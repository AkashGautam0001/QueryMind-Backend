from app.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def setup_tracing(app) -> None:
    settings = get_settings()
    if not settings.otlp_endpoint:
        logger.info("otel_tracing_disabled", reason="OTLP_ENDPOINT not configured")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": settings.app_name,
            "deployment.environment": settings.environment,
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel_tracing_enabled", endpoint=settings.otlp_endpoint)
    except ImportError:
        logger.warning(
            "otel_tracing_unavailable",
            reason=(
                "opentelemetry packages not installed. Add opentelemetry-sdk, "
                "opentelemetry-exporter-otlp, and opentelemetry-instrumentation-fastapi "
                "to pyproject.toml to enable tracing."
            ),
        )
