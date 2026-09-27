from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_PREDICATE_MODEL = "all-MiniLM-L6-v2"


def load_matcher_classes():
    try:
        from sg_eval.matchers import CLIPObjectMatcher, JinaPredicateMatcher, PredicateMatcher
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Live text query requires the sg_eval package on PYTHONPATH, plus "
            "its embedding dependencies. Static viewing with `python -m http.server` "
            "does not require sg_eval."
        ) from exc
    return CLIPObjectMatcher, JinaPredicateMatcher, PredicateMatcher


@dataclass
class QueryObject:
    id: int
    label: str


@dataclass
class QueryRelationship:
    index: int
    subject_id: int
    object_id: int
    predicate: str


def is_generic_object_label(label: str) -> bool:
    return str(label).strip().lower() == "object"


def normalize_query_text(text: str) -> str:
    return " ".join(str(text).lower().replace("_", " ").replace("-", " ").split())


def strip_leading_article(text: str) -> str:
    value = normalize_query_text(text)
    for article in ("the ", "a ", "an "):
        if value.startswith(article):
            return value[len(article):].strip()
    return value


RELATION_QUERY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("in front of", "front"),
    ("on top of", "on"),
    ("standing on", "standing on"),
    ("supported by", "supported by"),
    ("attached to", "attached to"),
    ("connected to", "connected to"),
    ("hanging on", "hanging on"),
    ("close by", "close by"),
    ("close to", "close by"),
    ("left of", "left"),
    ("right of", "right"),
    ("behind", "behind"),
    ("under", "under"),
    ("above", "above"),
    ("inside", "inside"),
    ("on", "on"),
    ("in", "in"),
)


def parse_relationship_query(text: str) -> tuple[str, str]:
    query = normalize_query_text(text)
    if not query:
        return "", ""
    for phrase, predicate in RELATION_QUERY_PREFIXES:
        prefix = f"{phrase} "
        if query.startswith(prefix):
            return predicate, strip_leading_article(query[len(prefix):])
    if " of " in query:
        predicate, anchor = query.rsplit(" of ", 1)
        return predicate.strip(), strip_leading_article(anchor)
    return query, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer-dir",
        type=Path,
        required=True,
        help="Generated PyViz3D viewer directory containing index.html and scene_query_objects.json.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host. Use 0.0.0.0 for the old login-node SSH tunnel.")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--query-top-k", type=int, default=5)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--predicate-matcher", choices=["sbert", "jina", "jina-v3"], default="sbert")
    parser.add_argument("--predicate-model", default=None)
    return parser.parse_args()


def load_query_payload(viewer_dir: Path) -> dict[str, Any]:
    path = viewer_dir / "scene_query_objects.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Regenerate the viewer first.")
    return json.loads(path.read_text(encoding="utf-8"))


class SceneQueryMatcher:
    def __init__(
        self,
        objects: list[QueryObject],
        relationships: list[QueryRelationship],
        clip_model: str,
        predicate_matcher_name: str,
        predicate_model: str,
    ):
        if not objects:
            raise ValueError("Cannot build a query matcher for an empty scene.")
        CLIPObjectMatcher, JinaPredicateMatcher, PredicateMatcher = load_matcher_classes()
        self.jina_predicate_matcher_cls = JinaPredicateMatcher
        self.predicate_matcher_cls = PredicateMatcher
        self.objects = objects
        self.object_by_id = {int(obj.id): obj for obj in self.objects}
        self.labels = [obj.label for obj in self.objects]
        self.matcher = CLIPObjectMatcher(clip_model)
        self.matcher.precompute(sorted(set(self.labels)))
        self.object_embeddings = self.matcher.get_embeddings(self.labels)
        self.object_embedding_by_id = {
            int(obj.id): self.object_embeddings[idx]
            for idx, obj in enumerate(self.objects)
        }

        self.relationships = [
            rel
            for rel in relationships
            if int(rel.subject_id) in self.object_by_id and int(rel.object_id) in self.object_by_id
        ]
        self.relationship_triplet_texts = [
            self.relationship_triplet_text(rel)
            for rel in self.relationships
        ]
        self.relationship_triplet_embeddings = None
        if self.relationship_triplet_texts:
            self.matcher.precompute(sorted(set(self.relationship_triplet_texts)))
            self.relationship_triplet_embeddings = self.matcher.get_embeddings(self.relationship_triplet_texts)
        self.predicate_matcher_name = predicate_matcher_name
        self.predicate_model = predicate_model
        self.predicate_matcher = None
        self.relationship_predicates = [rel.predicate for rel in self.relationships]
        self.relationship_predicate_embeddings = None

    def relationship_triplet_text(self, rel: QueryRelationship) -> str:
        subject = self.object_by_id[int(rel.subject_id)]
        target = self.object_by_id[int(rel.object_id)]
        return normalize_query_text(f"{subject.label} {rel.predicate} {target.label}")

    def ensure_predicate_matcher(self) -> None:
        if self.predicate_matcher is not None:
            return
        if self.predicate_matcher_name in {"jina", "jina-v3"}:
            self.predicate_matcher = self.jina_predicate_matcher_cls(self.predicate_model)
        else:
            self.predicate_matcher = self.predicate_matcher_cls(self.predicate_model)
        if self.relationship_predicates:
            self.predicate_matcher.precompute(sorted(set(self.relationship_predicates)))
            self.relationship_predicate_embeddings = self.predicate_matcher.get_embeddings(self.relationship_predicates)
        else:
            self.relationship_predicate_embeddings = np.zeros((0, 1), dtype=np.float32)

    def query(self, text: str, top_k: int) -> dict[str, Any]:
        query_text = text.strip()
        if not query_text:
            return {"query": text, "matches": [], "matched_ids": []}

        query_embedding = self.matcher.encode(query_text)
        similarities = np.asarray(self.object_embeddings @ query_embedding, dtype=np.float32)
        order = np.argsort(-similarities)
        specific_order = np.asarray(
            [idx for idx in order if not is_generic_object_label(self.objects[int(idx)].label)],
            dtype=order.dtype,
        )
        if len(specific_order):
            order = specific_order
        top_k = max(1, min(int(top_k), len(order)))

        matches = []
        for idx in order[:top_k]:
            obj = self.objects[int(idx)]
            matches.append(
                {
                    "id": int(obj.id),
                    "label": obj.label,
                    "similarity": float(similarities[int(idx)]),
                }
            )

        best = matches[0]
        best_label = best["label"]
        matched_ids = [
            int(obj.id)
            for obj, score in zip(self.objects, similarities)
            if obj.label == best_label and abs(float(score) - float(best["similarity"])) < 1e-6
        ]
        return {
            "query": query_text,
            "best": best,
            "matches": matches,
            "matched_ids": matched_ids or [int(best["id"])],
        }

    def relationship_query(
        self,
        text: str,
        top_k: int,
        subject_text: str = "",
        predicate_text: str = "",
    ) -> dict[str, Any]:
        query_text = text.strip()
        if not self.relationships:
            return {"query": query_text, "matches": [], "matched_ids": [], "matched_relationship_indices": []}

        self.ensure_predicate_matcher()
        subject_text = strip_leading_article(subject_text)
        predicate_text = normalize_query_text(predicate_text)
        if not subject_text and not predicate_text:
            predicate_text, subject_text = parse_relationship_query(query_text)
        if not subject_text and not predicate_text:
            return {"query": text, "matches": [], "matched_ids": [], "matched_relationship_indices": []}

        predicate_embedding = self.predicate_matcher.encode(predicate_text) if predicate_text else None
        subject_embedding = self.matcher.encode(subject_text) if subject_text else None

        scored_matches = []
        for rel_idx, rel in enumerate(self.relationships):
            subject = self.object_by_id[int(rel.subject_id)]
            target = self.object_by_id[int(rel.object_id)]
            predicate_similarity = 1.0
            subject_similarity = 1.0
            if predicate_embedding is not None:
                predicate_similarity = float(self.relationship_predicate_embeddings[rel_idx] @ predicate_embedding)
            if subject_embedding is not None:
                subject_similarity = float(self.object_embedding_by_id[int(subject.id)] @ subject_embedding)
            score = float(predicate_similarity * subject_similarity)
            scored_matches.append(
                {
                    "relationship_index": int(rel.index),
                    "subject_id": int(subject.id),
                    "subject_label": subject.label,
                    "predicate": rel.predicate,
                    "object_id": int(target.id),
                    "object_label": target.label,
                    "score": score,
                    "predicate_similarity": predicate_similarity,
                    "subject_similarity": subject_similarity,
                }
            )

        scored_matches.sort(key=lambda item: item["score"], reverse=True)
        specific_matches = [
            item
            for item in scored_matches
            if not is_generic_object_label(item["subject_label"]) and not is_generic_object_label(item["object_label"])
        ]
        if specific_matches:
            scored_matches = specific_matches
        top_k = max(1, min(int(top_k), len(scored_matches)))
        matches = scored_matches[:top_k]
        best_score = float(matches[0]["score"]) if matches else -float("inf")
        selected_matches = [
            match
            for match in matches
            if abs(float(match["score"]) - best_score) < 1e-6
        ]
        matched_ids = []
        context_ids = []
        rel_indices = []
        for match in selected_matches:
            if match["object_id"] not in matched_ids:
                matched_ids.append(match["object_id"])
            if match["subject_id"] not in context_ids:
                context_ids.append(match["subject_id"])
            rel_indices.append(match["relationship_index"])
        return {
            "query": query_text,
            "parsed": {
                "predicate": predicate_text,
                "subject": subject_text,
            },
            "best": matches[0] if matches else None,
            "matches": matches,
            "matched_ids": matched_ids,
            "context_ids": context_ids,
            "matched_relationship_indices": rel_indices,
        }

    def triplet_query(self, text: str, top_k: int, highlight_role: str = "object") -> dict[str, Any]:
        query_text = normalize_query_text(text)
        if not query_text or not self.relationships:
            return {
                "query": query_text,
                "highlight_role": "object" if highlight_role != "subject" else "subject",
                "matches": [],
                "matched_ids": [],
                "highlighted_ids": [],
                "matched_relationship_indices": [],
            }

        role = "subject" if highlight_role == "subject" else "object"
        if self.relationship_triplet_embeddings is None:
            self.relationship_triplet_embeddings = self.matcher.get_embeddings(self.relationship_triplet_texts)

        query_embedding = self.matcher.encode(query_text)
        similarities = np.asarray(self.relationship_triplet_embeddings @ query_embedding, dtype=np.float32)
        order = np.argsort(-similarities)
        top_k = max(1, min(int(top_k), len(order)))

        matches = []
        for idx in order[:top_k]:
            rel = self.relationships[int(idx)]
            subject = self.object_by_id[int(rel.subject_id)]
            target = self.object_by_id[int(rel.object_id)]
            matches.append(
                {
                    "relationship_index": int(rel.index),
                    "subject_id": int(subject.id),
                    "subject_label": subject.label,
                    "predicate": rel.predicate,
                    "object_id": int(target.id),
                    "object_label": target.label,
                    "triplet_text": self.relationship_triplet_texts[int(idx)],
                    "similarity": float(similarities[int(idx)]),
                }
            )

        best = matches[0] if matches else None
        if best is None:
            return {
                "query": query_text,
                "highlight_role": role,
                "matches": [],
                "matched_ids": [],
                "highlighted_ids": [],
                "matched_relationship_indices": [],
            }

        best_score = float(best["similarity"])
        selected_matches = [
            match
            for match in matches
            if abs(float(match["similarity"]) - best_score) < 1e-6
        ]
        highlighted_ids = []
        context_ids = []
        relationship_indices = []
        highlight_key = f"{role}_id"
        context_key = "subject_id" if role == "object" else "object_id"
        for match in selected_matches:
            if match[highlight_key] not in highlighted_ids:
                highlighted_ids.append(match[highlight_key])
            if match[context_key] not in context_ids:
                context_ids.append(match[context_key])
            relationship_indices.append(match["relationship_index"])

        return {
            "query": query_text,
            "highlight_role": role,
            "best": best,
            "matches": matches,
            "matched_ids": highlighted_ids,
            "highlighted_ids": highlighted_ids,
            "context_ids": context_ids,
            "matched_relationship_indices": relationship_indices,
        }


def build_matcher(args: argparse.Namespace) -> SceneQueryMatcher:
    payload = load_query_payload(args.viewer_dir)
    matcher_config = payload.get("matcher", {})
    clip_model = args.clip_model or matcher_config.get("clip_model") or DEFAULT_CLIP_MODEL
    predicate_model = args.predicate_model
    if predicate_model is None:
        if args.predicate_matcher == "jina-v3":
            predicate_model = "jinaai/jina-embeddings-v3"
        elif args.predicate_matcher == "jina":
            predicate_model = "jinaai/jina-embeddings-v2-base-en"
        else:
            predicate_model = DEFAULT_PREDICATE_MODEL
    objects = [
        QueryObject(id=int(obj["id"]), label=str(obj["label"]).lower().strip())
        for obj in payload.get("objects", [])
    ]
    relationships = [
        QueryRelationship(
            index=int(rel["index"]),
            subject_id=int(rel["subject_id"]),
            object_id=int(rel["object_id"]),
            predicate=str(rel["predicate"]).lower().strip(),
        )
        for rel in payload.get("relationships", [])
    ]
    return SceneQueryMatcher(
        objects,
        relationships,
        clip_model,
        args.predicate_matcher,
        predicate_model,
    )


def serve(args: argparse.Namespace) -> None:
    viewer_dir = args.viewer_dir.resolve()
    matcher = build_matcher(args)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(viewer_dir), **handler_kwargs)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path not in {"/scene-query", "/scene-relationship-query", "/scene-triplet-query"}:
                super().do_GET()
                return

            params = urllib.parse.parse_qs(parsed.query)
            text = params.get("text", [""])[0]
            subject = params.get("subject", [""])[0]
            predicate = params.get("predicate", [""])[0]
            highlight_role = params.get("highlight", params.get("highlight_role", ["object"]))[0]
            top_k = int(params.get("top_k", [args.query_top_k])[0])
            try:
                if parsed.path == "/scene-triplet-query":
                    payload = matcher.triplet_query(text, top_k, highlight_role=highlight_role)
                elif parsed.path == "/scene-relationship-query":
                    payload = matcher.relationship_query(text, top_k, subject_text=subject, predicate_text=predicate)
                else:
                    payload = matcher.query(text, top_k)
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = str(exc).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = ThreadingHTTPServer((str(args.host), int(args.port)), Handler)
    display_host = "127.0.0.1" if str(args.host) in {"0.0.0.0", "::"} else str(args.host)
    print(f"Serving viewer with semantic query at http://{display_host}:{args.port}/index.html")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped viewer server.")
    finally:
        server.server_close()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
