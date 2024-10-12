from datetime import datetime
from dotenv import load_dotenv
from hashlib import md5
from loguru import logger
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag.base import BaseKVStorage
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from shutil import rmtree
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from time import time
import argparse
import logging
import numpy as np
import os
import sys


logger.remove()
logger.add(sys.stdout, level="INFO")

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_{current_time}.log"
logger.add(log_filename, level="INFO")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0)

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
CHAT_MODEL = "deepseek-chat"
EMBEDDING_MODEL = "BAAI/bge-m3"

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


def clear_directory(directory):
    if not os.path.exists(directory):
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            rmtree(file_path)


class GraphRAGHandler:
    def __init__(self, document, working_dir=None):
        self.document = document

        if working_dir is None:
            self.working_dir = f"cache/{md5(document.encode()).hexdigest()}"
        else:
            self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        self.rag = GraphRAG(
            working_dir=self.working_dir,
            embedding_func=openapi_embedding,
            best_model_func=openapi_model_if_cache,
            cheap_model_func=openapi_model_if_cache,
        )

    def query(self, user_input):
        return self.rag.query(user_input, param=QueryParam(mode="global"))

    def insert(self):
        clear_directory(self.working_dir)
        start = time()
        with open(self.document, encoding="utf-8-sig") as f:
            self.rag.insert(f.read())
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

    handler = GraphRAGHandler(document=args.file)
    if args.operation == "insert":
        handler.insert()
    elif args.operation == "query":
        # "What are the top themes in this story?"
        while True:
            user_input = input("Enter your query (or type 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            logger.info(handler.query(user_input))


if __name__ == "__main__":
    main()
