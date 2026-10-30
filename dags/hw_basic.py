from typing import Any

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, task

from batch_common import (
    business_date,
    digest,
    infer_batches,
    read,
    replace_day,
    save,
    validate_predictions,
    validate_rows,
)

TZ = "Europe/Moscow"


@dag(
    dag_id="hw_basic_batch_inference",
    start_date=pendulum.datetime(2026, 9, 1, tz=TZ),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),
    },
    default_args={
        "retries": 2,
        "retry_delay": pendulum.duration(seconds=5),
        "execution_timeout": pendulum.duration(minutes=2),
    },
    tags=["homework", "basic"],
)
def hw_basic_batch_inference():
    @task
    def extract(**context: Any) -> dict:
        day = business_date(
            context["params"]["business_date"],
            context.get("data_interval_start"),
        )
        host = BaseHook.get_connection("source_api").host.rstrip("/")

        rows_response = requests.get(f"{host}/basic/{day}.json", timeout=10)
        rows_response.raise_for_status()
        rows_payload = rows_response.json()
        if rows_payload["business_date"] != day:
            raise ValueError("Source returned a different business_date")

        model_response = requests.get(f"{host}/models/v1.json", timeout=10)
        model_response.raise_for_status()

        return save({
            "business_date": day,
            "rows": rows_payload["rows"],
            "model": model_response.json(),
            "model_sha256": digest(model_response.content),
        })

    @task(retries=0)
    def infer(extract_ref: dict, **context: Any) -> dict:
        payload = read(extract_ref)
        rows = validate_rows(payload["rows"], payload["business_date"])
        predictions = infer_batches(
            rows,
            payload["model"],
            context["params"]["batch_size"],
        )
        validate_predictions(
            predictions,
            [row["customer_id"] for row in rows],
        )

        return save({
            "business_date": payload["business_date"],
            "model_version": "v1",
            "predictions": predictions,
            "model_sha256": payload["model_sha256"],
        })

    @task
    def publish(infer_ref: dict) -> dict:
        payload = read(infer_ref)
        connection = PostgresHook(postgres_conn_id="warehouse").get_conn()
        try:
            replace_day(
                connection,
                payload["business_date"],
                payload["model_version"],
                payload["predictions"],
            )
        finally:
            connection.close()

        return {
            "business_date": payload["business_date"],
            "model_version": payload["model_version"],
            "rows": len(payload["predictions"]),
        }

    extracted = extract()
    inferred = infer(extracted)
    publish(inferred)


hw_basic_batch_inference()
