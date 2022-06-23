import aiohttp
import requests
from loguru import logger


URL = "http://0.0.0.0:5000/predict/random"


async def concurrent():  # FIXME: RuntimeWarning: coroutine 'concurrent' was never awaited concurrent()
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                r = await session.get(URL)
                logger.info(await r.json())
            except ValueError as e:
                logger.warning(e)


def synchronous():
    while True:
        try:
            r = requests.get(URL)
            logger.info(r.json())
        except ValueError as e:
            logger.warning(e)


if __name__ == "__main__":
    synchronous()
