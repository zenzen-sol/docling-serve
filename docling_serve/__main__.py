import importlib.metadata
import json
import logging
import os
import platform
import sys
import warnings
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Optional, Union

import requests
import typer
import uvicorn
from google.cloud import storage
from rich.console import Console

from docling.datamodel.base_models import DocumentStream

from docling_serve.datamodel.convert import ConvertDocumentsOptions
from docling_serve.docling_conversion import convert_documents
from docling_serve.settings import docling_serve_settings, uvicorn_settings

warnings.filterwarnings(action="ignore", category=UserWarning, module="pydantic|torch")
warnings.filterwarnings(action="ignore", category=FutureWarning, module="easyocr")


err_console = Console(stderr=True)
console = Console()

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
)

logger = logging.getLogger(__name__)


def version_callback(value: bool) -> None:
    if value:
        docling_serve_version = importlib.metadata.version("docling_serve")
        docling_version = importlib.metadata.version("docling")
        docling_core_version = importlib.metadata.version("docling-core")
        docling_ibm_models_version = importlib.metadata.version("docling-ibm-models")
        docling_parse_version = importlib.metadata.version("docling-parse")
        platform_str = platform.platform()
        py_impl_version = sys.implementation.cache_tag
        py_lang_version = platform.python_version()
        console.print(f"Docling Serve version: {docling_serve_version}")
        console.print(f"Docling version: {docling_version}")
        console.print(f"Docling Core version: {docling_core_version}")
        console.print(f"Docling IBM Models version: {docling_ibm_models_version}")
        console.print(f"Docling Parse version: {docling_parse_version}")
        console.print(f"Python: {py_impl_version} ({py_lang_version})")
        console.print(f"Platform: {platform_str}")
        raise typer.Exit()


@app.callback()
def callback(
    version: Annotated[
        Union[bool, None],
        typer.Option(help="Show the version and exit.", callback=version_callback),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Set the verbosity level. -v for info logging, -vv for debug logging.",
        ),
    ] = 0,
    mode: Annotated[
        str, typer.Option(help="The mode to run in. Can be 'server' or 'job'.")
    ] = "server",
    file_uri: Annotated[
        Optional[str], typer.Option(help="The URI of the file to process in job mode.")
    ] = None,
    job_id: Annotated[
        Optional[str], typer.Option(help="The ID of the job to process in job mode.")
    ] = None,
) -> None:
    if verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 2:
        logging.basicConfig(level=logging.DEBUG)

    docling_serve_settings.mode = mode
    docling_serve_settings.file_uri = file_uri
    docling_serve_settings.job_id = job_id


def run_job(file_uri: str, job_id: str) -> None:
    """Runs the document processing job."""
    console.print(f"Running job {job_id} for file {file_uri}")

    try:
        # 1. Download the file from the URI
        console.print(f"Downloading file from {file_uri}...")
        response = requests.get(file_uri)
        response.raise_for_status()
        file_content = BytesIO(response.content)
        console.print("File downloaded successfully.")

        # 2. Process the document using the docling pipeline
        console.print("Starting document conversion...")
        task_source = DocumentStream(name=job_id, stream=file_content)
        options = ConvertDocumentsOptions(
            to_formats=["json"],
            include_images=True,
            pipeline="vlm",
            ocr_engine="easyocr",
            ocr_lang=["en"],
            image_export_mode="placeholder",
        )

        # This is a synchronous call that will do the heavy lifting
        results = convert_documents(sources=[task_source], options=options)
        console.print("Document conversion completed.")

        # Assuming single document processing for now
        if not results or not results[0].json_content:
            raise Exception("Conversion did not produce any JSON content.")

        result_json = results[0].json_content.model_dump_json(indent=2)

        # 3. Save the results to GCS
        output_bucket_name = os.getenv("DOCLING_JOB_OUTPUT_BUCKET")
        if not output_bucket_name:
            raise ValueError(
                "DOCLING_JOB_OUTPUT_BUCKET environment variable must be set"
            )

        console.print(f"Uploading result to gs://{output_bucket_name}/{job_id}.json")
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket_name.replace("gs://", ""))
        blob = bucket.blob(f"{job_id}.json")
        blob.upload_from_string(result_json, content_type="application/json")
        console.print("Result uploaded successfully.")

    except Exception as e:
        console.print(f"[bold red]Error in job {job_id}:[/bold red] {e}")
        # Re-raise the exception to make the Cloud Run job fail
        raise

    console.print(f"Job {job_id} completed.")


def _run(
    *,
    command: str,
    # Docling serve parameters
    artifacts_path: Path | None,
    enable_ui: bool,
) -> None:
    if docling_serve_settings.mode == "job":
        if not docling_serve_settings.file_uri or not docling_serve_settings.job_id:
            err_console.print(
                "[bold red]Error:[/bold red] --file-uri and --job-id are required in job mode."
            )
            raise typer.Exit(1)
        run_job(docling_serve_settings.file_uri, docling_serve_settings.job_id)
        return

    server_type = "development" if command == "dev" else "production"

    console.print(f"Starting {server_type} server 🚀")

    run_subprocess = (
        uvicorn_settings.workers is not None and uvicorn_settings.workers > 1
    ) or uvicorn_settings.reload

    run_ssl = (
        uvicorn_settings.ssl_certfile is not None
        and uvicorn_settings.ssl_keyfile is not None
    )

    if run_subprocess and docling_serve_settings.artifacts_path != artifacts_path:
        err_console.print(
            "\n[yellow]:warning: The server will run with reload or multiple workers. \n"
            "The argument [bold]--artifacts-path[/bold] will be ignored, please set the value \n"
            "using the environment variable [bold]DOCLING_SERVE_ARTIFACTS_PATH[/bold].[/yellow]"
        )

    if run_subprocess and docling_serve_settings.enable_ui != enable_ui:
        err_console.print(
            "\n[yellow]:warning: The server will run with reload or multiple workers. \n"
            "The argument [bold]--enable-ui[/bold] will be ignored, please set the value \n"
            "using the environment variable [bold]DOCLING_SERVE_ENABLE_UI[/bold].[/yellow]"
        )

    # Propagate the settings to the app settings
    docling_serve_settings.artifacts_path = artifacts_path
    docling_serve_settings.enable_ui = enable_ui

    # Print documentation
    protocol = "https" if run_ssl else "http"
    url = f"{protocol}://{uvicorn_settings.host}:{uvicorn_settings.port}"
    url_docs = f"{url}/docs"
    url_ui = f"{url}/ui"

    console.print("")
    console.print(f"Server started at [link={url}]{url}[/]")
    console.print(f"Documentation at [link={url_docs}]{url_docs}[/]")
    if docling_serve_settings.enable_ui:
        console.print(f"UI at [link={url_ui}]{url_ui}[/]")

    if command == "dev":
        console.print("")
        console.print(
            "Running in development mode, for production use: "
            "[bold]docling-serve run[/]",
        )

    console.print("")
    console.print("Logs:")

    # Launch the server
    uvicorn.run(
        app="docling_serve.app:create_app",
        factory=True,
        host=uvicorn_settings.host,
        port=uvicorn_settings.port,
        reload=uvicorn_settings.reload,
        workers=uvicorn_settings.workers,
        root_path=uvicorn_settings.root_path,
        proxy_headers=uvicorn_settings.proxy_headers,
        timeout_keep_alive=uvicorn_settings.timeout_keep_alive,
        ssl_certfile=uvicorn_settings.ssl_certfile,
        ssl_keyfile=uvicorn_settings.ssl_keyfile,
        ssl_keyfile_password=uvicorn_settings.ssl_keyfile_password,
    )


@app.command()
def dev(
    *,
    # uvicorn options
    host: Annotated[
        str,
        typer.Option(
            help=(
                "The host to serve on. For local development in localhost "
                "use [blue]127.0.0.1[/blue]. To enable public access, "
                "e.g. in a container, use all the IP addresses "
                "available with [blue]0.0.0.0[/blue]."
            )
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(help="The port to serve on."),
    ] = uvicorn_settings.port,
    reload: Annotated[
        bool,
        typer.Option(
            help=(
                "Enable auto-reload of the server when (code) files change. "
                "This is [bold]resource intensive[/bold], "
                "use it only during development."
            )
        ),
    ] = True,
    root_path: Annotated[
        str,
        typer.Option(
            help=(
                "The root path is used to tell your app that it is being served "
                "to the outside world with some [bold]path prefix[/bold] "
                "set up in some termination proxy or similar."
            )
        ),
    ] = uvicorn_settings.root_path,
    proxy_headers: Annotated[
        bool,
        typer.Option(
            help=(
                "Enable/Disable X-Forwarded-Proto, X-Forwarded-For, "
                "X-Forwarded-Port to populate remote address info."
            )
        ),
    ] = uvicorn_settings.proxy_headers,
    timeout_keep_alive: Annotated[
        int, typer.Option(help="Timeout for the server response.")
    ] = uvicorn_settings.timeout_keep_alive,
    ssl_certfile: Annotated[
        Optional[Path], typer.Option(help="SSL certificate file")
    ] = uvicorn_settings.ssl_certfile,
    ssl_keyfile: Annotated[
        Optional[Path], typer.Option(help="SSL key file")
    ] = uvicorn_settings.ssl_keyfile,
    ssl_keyfile_password: Annotated[
        Optional[str], typer.Option(help="SSL keyfile password")
    ] = uvicorn_settings.ssl_keyfile_password,
    # docling options
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "If set to a valid directory, "
                "the model weights will be loaded from this path."
            )
        ),
    ] = docling_serve_settings.artifacts_path,
    enable_ui: Annotated[bool, typer.Option(help="Enable the development UI.")] = True,
) -> Any:
    """
    Run a [bold]Docling Serve[/bold] app in [yellow]development[/yellow] mode. 🧪

    This is equivalent to [bold]docling-serve run[/bold] but with [bold]reload[/bold]
    enabled and listening on the [blue]127.0.0.1[/blue] address.

    Options can be set also with the corresponding ENV variable, with the exception
    of --enable-ui, --host and --reload.
    """

    uvicorn_settings.host = host
    uvicorn_settings.port = port
    uvicorn_settings.reload = reload
    uvicorn_settings.root_path = root_path
    uvicorn_settings.proxy_headers = proxy_headers
    uvicorn_settings.timeout_keep_alive = timeout_keep_alive
    uvicorn_settings.ssl_certfile = ssl_certfile
    uvicorn_settings.ssl_keyfile = ssl_keyfile
    uvicorn_settings.ssl_keyfile_password = ssl_keyfile_password

    _run(
        command="dev",
        artifacts_path=artifacts_path,
        enable_ui=enable_ui,
    )


@app.command()
def run(
    *,
    # uvicorn options
    host: Annotated[
        str,
        typer.Option(
            help=(
                "The host to serve on. For local development in localhost "
                "use [blue]127.0.0.1[/blue]. To enable public access, "
                "e.g. in a container, use all the IP addresses "
                "available with [blue]0.0.0.0[/blue]."
            )
        ),
    ] = uvicorn_settings.host,
    port: Annotated[
        int,
        typer.Option(help="The port to serve on."),
    ] = uvicorn_settings.port,
    reload: Annotated[
        bool,
        typer.Option(
            help=(
                "Enable auto-reload of the server when (code) files change. "
                "This is [bold]resource intensive[/bold], "
                "use it only during development."
            )
        ),
    ] = uvicorn_settings.reload,
    workers: Annotated[
        Union[int, None],
        typer.Option(
            help=(
                "Use multiple worker processes. "
                "Mutually exclusive with the --reload flag."
            )
        ),
    ] = uvicorn_settings.workers,
    root_path: Annotated[
        str,
        typer.Option(
            help=(
                "The root path is used to tell your app that it is being served "
                "to the outside world with some [bold]path prefix[/bold] "
                "set up in some termination proxy or similar."
            )
        ),
    ] = uvicorn_settings.root_path,
    proxy_headers: Annotated[
        bool,
        typer.Option(
            help=(
                "Enable/Disable X-Forwarded-Proto, X-Forwarded-For, "
                "X-Forwarded-Port to populate remote address info."
            )
        ),
    ] = uvicorn_settings.proxy_headers,
    timeout_keep_alive: Annotated[
        int, typer.Option(help="Timeout for the server response.")
    ] = uvicorn_settings.timeout_keep_alive,
    ssl_certfile: Annotated[
        Optional[Path], typer.Option(help="SSL certificate file")
    ] = uvicorn_settings.ssl_certfile,
    ssl_keyfile: Annotated[
        Optional[Path], typer.Option(help="SSL key file")
    ] = uvicorn_settings.ssl_keyfile,
    ssl_keyfile_password: Annotated[
        Optional[str], typer.Option(help="SSL keyfile password")
    ] = uvicorn_settings.ssl_keyfile_password,
    # docling options
    artifacts_path: Annotated[
        Optional[Path],
        typer.Option(
            help=(
                "If set to a valid directory, "
                "the model weights will be loaded from this path."
            )
        ),
    ] = docling_serve_settings.artifacts_path,
    enable_ui: Annotated[
        bool, typer.Option(help="Enable the development UI.")
    ] = docling_serve_settings.enable_ui,
) -> Any:
    """
    Run a [bold]Docling Serve[/bold] app in [green]production[/green] mode. 🥦

    This is the recommended way to run the app in a production environment.

    Options can be set also with the corresponding ENV variable.
    """

    uvicorn_settings.host = host
    uvicorn_settings.port = port
    uvicorn_settings.reload = reload
    uvicorn_settings.workers = workers
    uvicorn_settings.root_path = root_path
    uvicorn_settings.proxy_headers = proxy_headers
    uvicorn_settings.timeout_keep_alive = timeout_keep_alive
    uvicorn_settings.ssl_certfile = ssl_certfile
    uvicorn_settings.ssl_keyfile = ssl_keyfile
    uvicorn_settings.ssl_keyfile_password = ssl_keyfile_password

    _run(
        command="run",
        artifacts_path=artifacts_path,
        enable_ui=enable_ui,
    )


def main() -> None:
    # This is a hack to make the app work with the IDE
    # https://github.com/tiangolo/typer/issues/152
    if sys.gettrace() is not None:
        sys.argv.append("dev")

    app()


if __name__ == "__main__":
    main()
