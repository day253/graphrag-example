import os
import logging
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from loguru import logger
import sys

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import argparse
from shutil import rmtree

# Set the logger to output only INFO level and above
logger.remove()  # Remove the default logger configuration
logger.add(sys.stdout, level="INFO")
# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


# Define a function to redirect standard logging to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the log message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


# Set up the logging configuration to use the InterceptHandler
logging.basicConfig(handlers=[InterceptHandler()], level=0)

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_MODEL = "deepseek-chat"
EMBEDDING_MODEL = "BAAI/bge-m3"
WORKING_DIR = "./nano_graphrag_cache_deepseek_TEST"

siliconflow_async_client = AsyncOpenAI(
    api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1/"
)
deepseek_async_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1/"
)
global_chat_async_client = deepseek_async_client
global_embedding_async_client = siliconflow_async_client


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openapi_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = global_embedding_async_client
    response = await openai_async_client.embeddings.create(
        model=EMBEDDING_MODEL, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def openapi_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = global_chat_async_client
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(CHAT_MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=CHAT_MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": response.choices[0].message.content,
                    "model": CHAT_MODEL,
                }
            }
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        embedding_func=openapi_embedding,
        best_model_func=openapi_model_if_cache,
        cheap_model_func=openapi_model_if_cache,
    )
    # "What are the top themes in this story?"
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        logger.info(rag.query(user_input, param=QueryParam(mode="global")))


def insert(file_path):
    from time import time

    clear_directory(f"{WORKING_DIR}")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        embedding_func=openapi_embedding,
        best_model_func=openapi_model_if_cache,
        cheap_model_func=openapi_model_if_cache,
    )
    start = time()
    with open(file_path, encoding="utf-8-sig") as f:
        rag.insert(f.read())
    logger.info(f"indexing time: {time() - start}")


def main():
    parser = argparse.ArgumentParser(description="Run insert or query operations.")
    parser.add_argument(
        "operation",
        choices=["insert", "query"],
        help="Specify the operation to perform: 'insert' or 'query'.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File path for the insert operation.",
    )
    args = parser.parse_args()
    logging.info(args)

    if args.operation == "insert":
        insert(args.file)
    elif args.operation == "query":
        query()


if __name__ == "__main__":
    main()
