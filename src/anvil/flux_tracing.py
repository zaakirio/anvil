"""Optional, env-gated OpenTelemetry tracing that ships spans to flux as OTLP/HTTP JSON.

Nothing here runs unless FLUX_OTLP_ENDPOINT is set AND the `a2a` extra is installed
(`uv sync --extra a2a`). The base CLI/MCP paths and the 41-test suite never import this.

flux ingests OTLP/HTTP JSON, so this uses a small JSON exporter rather than the
proto-http exporter. The spans themselves are genuine OpenTelemetry SDK spans with real
timing, context, and parent linkage; only the wire encoding is hand-rolled.
"""

import json
import os
import urllib.request

_SETUP_DONE = False


def tracing_enabled() -> bool:
    return bool(os.environ.get("FLUX_OTLP_ENDPOINT"))


def _attr_value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _make_exporter(endpoint: str):
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class FluxJsonSpanExporter(SpanExporter):
        def export(self, spans) -> "SpanExportResult":
            if not spans:
                return SpanExportResult.SUCCESS
            resource = spans[0].resource
            out_spans = []
            for s in spans:
                ctx = s.get_span_context()
                span_json = {
                    "traceId": format(ctx.trace_id, "032x"),
                    "spanId": format(ctx.span_id, "016x"),
                    "name": s.name,
                    "kind": s.kind.value,
                    "startTimeUnixNano": str(s.start_time),
                    "endTimeUnixNano": str(s.end_time),
                    "attributes": [
                        {"key": k, "value": _attr_value(v)} for k, v in s.attributes.items()
                    ],
                    "status": {"code": s.status.status_code.value},
                }
                if s.parent is not None:
                    span_json["parentSpanId"] = format(s.parent.span_id, "016x")
                out_spans.append(span_json)
            payload = {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": k, "value": _attr_value(v)}
                                for k, v in resource.attributes.items()
                            ]
                        },
                        "scopeSpans": [{"scope": {"name": "anvil"}, "spans": out_spans}],
                    }
                ]
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint, data=data, headers={"content-type": "application/json"}
            )
            try:
                urllib.request.urlopen(req, timeout=5).read()
                return SpanExportResult.SUCCESS
            except Exception:
                return SpanExportResult.FAILURE

        def shutdown(self) -> None:
            pass

    return FluxJsonSpanExporter()


def setup_tracing(service_name: str = "anvil"):
    """Idempotently install a TracerProvider that exports to flux; return a tracer."""
    global _SETUP_DONE
    from opentelemetry import trace

    if _SETUP_DONE or not tracing_enabled():
        return trace.get_tracer(service_name)

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = os.environ["FLUX_OTLP_ENDPOINT"]
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(SimpleSpanProcessor(_make_exporter(endpoint)))
    trace.set_tracer_provider(provider)
    _SETUP_DONE = True
    return trace.get_tracer(service_name)


def extract_context(headers: dict):
    """Turn an incoming header dict (e.g. carrying W3C traceparent) into an OTel Context."""
    from opentelemetry.propagate import extract

    return extract({k.lower(): v for k, v in headers.items()})
