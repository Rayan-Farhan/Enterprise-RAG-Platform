"""Provision OpenSearch for the neural sparse channel (Task 6.2).

Idempotent — safe to re-run, and needed again after the OpenSearch data volume
is recreated:

    python -m scripts.setup_neural_sparse

Steps, each skipped when already done:

1. Cluster settings for a single-node development cluster: allow ML models on
   a data node, and stop ML Commons' native-memory breaker from refusing to
   load a model on a machine whose page cache makes memory look full. A
   production cluster runs dedicated ML nodes instead (ADR-016) and should not
   apply these — pass ``--no-dev-settings``.
2. Register the pretrained sparse *encoder* (runs at index time) and *tokenizer*
   (runs at query time). The first registration downloads the weights from
   artifacts.opensearch.org; the first deployment also downloads PyTorch's
   native libraries (~780 MB into the data volume's ml_cache).
3. Deploy both models.
4. Create the ingest pipeline that encodes ``content`` into ``content_sparse``,
   and the sparse index that uses it as its default pipeline.

Then encode the corpus with ``python -m scripts.index_lexical --sparse``.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.retrieval.sparse_store import OpenSearchSparseStore, find_models, get_sparse_store

DEV_CLUSTER_SETTINGS = {
    "persistent": {
        "plugins.ml_commons.only_run_on_ml_node": False,
        "plugins.ml_commons.native_memory_threshold": 100,
        "plugins.ml_commons.model_access_control_enabled": False,
    }
}
POLL_SECONDS = 5
TASK_TIMEOUT_SECONDS = 900


def _request(store: OpenSearchSparseStore, method: str, path: str, body: Any = None) -> Any:
    return store.client.transport.perform_request(method, path, body=body)


def _wait_for_task(store: OpenSearchSparseStore, task_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + TASK_TIMEOUT_SECONDS
    while True:
        task = _request(store, "GET", f"/_plugins/_ml/tasks/{task_id}")
        if task.get("state") not in {"CREATED", "RUNNING"}:
            return dict(task)
        if time.monotonic() > deadline:
            raise TimeoutError(f"ML task {task_id} still {task.get('state')} after timeout")
        time.sleep(POLL_SECONDS)


def _find_model(store: OpenSearchSparseStore, name: str) -> dict[str, Any] | None:
    """The registered model with this name, preferring one already deployed."""
    hits = find_models(store.client, name)
    if not hits:
        return None
    deployed = [h for h in hits if h["_source"].get("model_state") == "DEPLOYED"]
    chosen = (deployed or hits)[0]
    return {"id": chosen["_id"], **chosen["_source"]}


def ensure_model(store: OpenSearchSparseStore, name: str, version: str, function: str) -> str:
    """Register and deploy a pretrained model if needed; return its ID."""
    model = _find_model(store, name)
    if model is None:
        print(f"  registering {name}@{version} (downloads weights on first run)…")
        task = _request(
            store,
            "POST",
            "/_plugins/_ml/models/_register",
            {
                "name": name,
                "version": version,
                "model_format": "TORCH_SCRIPT",
                "function_name": function,
            },
        )
        done = _wait_for_task(store, task["task_id"])
        if done.get("state") != "COMPLETED":
            raise RuntimeError(f"Registering {name} failed: {done.get('error')}")
        model_id = str(done["model_id"])
        state = "REGISTERED"
    else:
        model_id, state = str(model["id"]), str(model.get("model_state"))

    if state != "DEPLOYED":
        # The first deployment on a fresh volume also fetches PyTorch; a memory
        # breaker can refuse the first attempt while that is still settling.
        for attempt in (1, 2):
            print(f"  deploying {name} (attempt {attempt})…")
            task = _request(store, "POST", f"/_plugins/_ml/models/{model_id}/_deploy")
            done = _wait_for_task(store, task["task_id"])
            if done.get("state") == "COMPLETED":
                break
            print(f"    deploy failed: {done.get('error')}")
        else:
            raise RuntimeError(f"Deploying {name} failed twice; see messages above")

    print(f"  {name}@{version}: DEPLOYED ({model_id})")
    return model_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="setup_neural_sparse", description=__doc__)
    parser.add_argument(
        "--no-dev-settings",
        action="store_true",
        help="do not apply the single-node ML cluster settings (production clusters)",
    )
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings()
    store = get_sparse_store()
    if not store.health_check():
        print("OpenSearch is not reachable; start it with `make up`.")
        return 1

    if not args.no_dev_settings:
        store.client.cluster.put_settings(body=DEV_CLUSTER_SETTINGS)
        print("Applied single-node ML cluster settings.")

    print("Models:")
    ensure_model(
        store, settings.SPARSE_DOC_MODEL, settings.SPARSE_DOC_MODEL_VERSION, "SPARSE_ENCODING"
    )
    ensure_model(
        store,
        settings.SPARSE_QUERY_TOKENIZER,
        settings.SPARSE_QUERY_TOKENIZER_VERSION,
        "SPARSE_TOKENIZE",
    )

    store.ensure_pipeline()
    print(f"Pipeline '{store.pipeline}' points at the deployed encoder.")
    created = store.ensure_index()
    print(f"Index '{store.index_name}' {'created' if created else 'already present'}.")
    print("\nNext: python -m scripts.index_lexical --sparse   (encodes the corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
