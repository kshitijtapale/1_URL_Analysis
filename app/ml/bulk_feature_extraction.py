import pandas as pd
from app.ml.feature_extraction import FeatureExtractor
from app.logger import setup_logger
from app.exceptions import FeatureExtractionError
from typing import List, Dict
import os
import asyncio
import aiohttp
import random

logger = setup_logger(__name__)


class BulkFeatureExtractor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.max_retries = 3
        self.base_delay = 1  # Base delay in seconds

    async def _check_google_index(self, session, url, semaphore):
        async with semaphore:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(f"http://google.com/search?q=site:{url}", allow_redirects=False) as response:
                        if response.status == 200:
                            return 1
                        elif response.status == 429:
                            delay = self.base_delay * \
                                (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limit hit for {
                                           url}. Retrying in {delay:.2f} seconds.")
                            await asyncio.sleep(delay)
                        else:
                            return 0
                except Exception as e:
                    logger.warning(f"Error checking Google index for {
                                   url}: {str(e)}")
                    await asyncio.sleep(1)
            logger.error(f"Failed to check Google index for {
                         url} after {self.max_retries} attempts.")
            return 0

    async def _bulk_google_index_check(self, urls):
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
            tasks = [self._check_google_index(
                session, url, semaphore) for url in urls]
            return await asyncio.gather(*tasks)

    async def extract_features_from_csv(self, input_file: str, output_file: str, check_google_index: bool = False) -> None:
        try:
            logger.info(f"Starting feature extraction from {input_file}")
            df = pd.read_csv(input_file)

            if 'url' not in df.columns or 'type' not in df.columns:
                raise ValueError(
                    "Input CSV must contain 'url' and 'type' columns")

            features: List[Dict[str, int]] = []
            for url in df['url']:
                features.append(self.feature_extractor.extract_features(url))

            features_df = pd.DataFrame(features)

            if check_google_index:
                logger.info("Starting Google indexing check")
                google_index_results = await self._bulk_google_index_check(df['url'].tolist())
                features_df['google_index'] = google_index_results
            else:
                features_df['google_index'] = -1  # -1 indicates not checked

            result_df = pd.concat([df, features_df], axis=1)

            result_df.to_csv(output_file, index=True, index_label='')
            logger.info(
                f"Feature extraction completed. Results saved to {output_file}")

        except Exception as e:
            logger.error(f"Error in bulk feature extraction: {str(e)}")
            raise FeatureExtractionError(
                f"Bulk feature extraction failed: {str(e)}")

"""     def generate_visualizations(self, input_file: str, output_dir: str) -> None:
        try:
            logger.info(f"Generating visualizations from {input_file}")
            df = pd.read_csv(input_file)

            import matplotlib.pyplot as plt
            import seaborn as sns
            from wordcloud import WordCloud

            os.makedirs(output_dir, exist_ok=True)

            # Distribution of Types of Attacks
            plt.figure(figsize=(10, 5))
            plt.title('Distribution of Types of Attacks')
            sns.countplot(x='type', data=df)
            plt.xlabel('Attacks')
            plt.savefig(os.path.join(
                output_dir, 'distribution_of_attacks.png'), bbox_inches='tight')
            plt.close()

            # Word Clouds
            for attack_type in df['type'].unique():
                urls = " ".join(df[df['type'] == attack_type]['url'])
                wordcloud = WordCloud(
                    width=1600, height=800, background_color="white", colormap="Set2").generate(urls)
                plt.figure(figsize=(16, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(output_dir, f'{
                            attack_type}_wordcloud.png'), bbox_inches='tight')
                plt.close()

            # Usage of IP Address
            plt.figure(figsize=(8, 6))
            sns.countplot(x="type", data=df, hue="use_of_ip")
            plt.xlabel('Types of Attacks')
            plt.title(
                "Usage of IP Address in domain name in Different types of attacks")
            plt.savefig(os.path.join(
                output_dir, 'ip_usage_by_attack_type.png'), bbox_inches='tight')
            plt.close()

            logger.info(f"Visualizations generated and saved in {output_dir}")

        except Exception as e:
            logger.error(f"Error in generating visualizations: {str(e)}")
            raise FeatureExtractionError(
                f"Visualization generation failed: {str(e)}")
 """