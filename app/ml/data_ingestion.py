import os
import pandas as pd
import asyncio
import aiofiles
from sklearn.model_selection import train_test_split
from app.config import settings
from app.exceptions import DataIngestionError
from app.logger import setup_logger
import glob
from typing import Tuple
import csv

logger = setup_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(settings.RAW_DATA_DIR, 'raw_data.csv')
        self.train_data_path = os.path.join(settings.TRAIN_DATA_DIR, 'train_data.csv')
        self.test_data_path = os.path.join(settings.TEST_DATA_DIR, 'test_data.csv')

    async def ensure_directories(self):
        dirs = [
            os.path.dirname(self.raw_data_path),
            os.path.dirname(self.train_data_path),
            os.path.dirname(self.test_data_path)
        ]
        await asyncio.gather(*[asyncio.to_thread(os.makedirs, d, exist_ok=True) for d in dirs])

    async def find_latest_file(self) -> str:
        pattern = os.path.join(settings.EXTRACTED_DATA, "preprocessed_*.csv")
        files = await asyncio.to_thread(glob.glob, pattern)
        if not files:
            raise FileNotFoundError(f"No preprocessed CSV files found in {settings.EXTRACTED_DATA}")
        return max(files, key=os.path.getctime)

    async def process_csv_in_chunks(self, input_file: str, output_train: str, output_test: str, chunksize: int = 10000):
        test_size = 0.2
        headers = None

        async with aiofiles.open(input_file, mode='r', encoding='utf-8', errors='replace', newline='') as infile, \
                   aiofiles.open(output_train, mode='w', encoding='utf-8', newline='') as train_file, \
                   aiofiles.open(output_test, mode='w', encoding='utf-8', newline='') as test_file:
            
            # Read and write headers
            headers = await infile.readline()
            await train_file.write(headers)
            await test_file.write(headers)

            chunk = []
            async for line in infile:
                try:
                    chunk.append(line.strip().split(','))
                    if len(chunk) >= chunksize:
                        await self.split_and_write_chunk(chunk, train_file, test_file, test_size)
                        chunk = []
                except Exception as e:
                    logger.warning(f"Skipping malformed line: {line.strip()}. Error: {str(e)}")

            if chunk:
                await self.split_and_write_chunk(chunk, train_file, test_file, test_size)

    async def split_and_write_chunk(self, chunk, train_file, test_file, test_size):
        train, test = train_test_split(chunk, test_size=test_size, random_state=42)
        await asyncio.gather(
            self.write_rows(train_file, train),
            self.write_rows(test_file, test)
        )

    async def write_rows(self, file, rows):
        for row in rows:
            await file.write(','.join(row) + '\n')

    async def initiate_data_ingestion(self) -> Tuple[str, str]:
        logger.info("Initiating data ingestion process.")
        try:
            await self.ensure_directories()

            latest_file = await self.find_latest_file()
            logger.info(f"Using preprocessed file: {latest_file}")

            await self.process_csv_in_chunks(latest_file, self.train_data_path, self.test_data_path)

            logger.info(f"Training data saved at {self.train_data_path}")
            logger.info(f"Testing data saved at {self.test_data_path}")
            logger.info("Data ingestion process completed successfully.")

            return self.train_data_path, self.test_data_path

        except Exception as e:
            logger.error(f"Exception occurred during Data Ingestion: {e}")
            raise DataIngestionError(f"Error occurred during data ingestion process: {str(e)}")

    @classmethod
    async def create_and_ingest(cls) -> Tuple[str, str]:
        ingestion = cls()
        return await ingestion.initiate_data_ingestion()