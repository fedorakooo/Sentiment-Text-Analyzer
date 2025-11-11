import logging
import asyncio

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from pydantic import ValidationError

from src.exceptions import SentimentAnalysisError
from src.models import SentimentResult
from src.cache import RedisCache

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    def __init__(
            self,
            cache: RedisCache,
            ollama_base_url: str = "http://localhost:11434",
            ollama_model: str = "llama3",
            request_timeout: int = 30
    ):
        self.cache = cache
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.request_timeout = request_timeout

        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=ollama_model,
        )
        self.prompt = self._create_prompt()
        self.parser = JsonOutputParser()

    def _create_prompt(self) -> ChatPromptTemplate:
        template = """
        You are an expert sentiment analysis assistant. Analyze the sentiment of the provided text and return a strict JSON object with the following fields:
    
        - "label": one of "positive", "negative", "neutral", or "mixed"
        - "confidence": a float between 0 and 1 indicating your confidence level
        - "explanation": a concise explanation of why you chose this label (max 20 words)

        IMPORTANT RULES:
        1. Respond **ONLY** with a single valid JSON object.
        2. Do **NOT** include any text, greetings, or instructions outside the JSON.
        3. Make sure the JSON is always properly formatted, with correct quotes and commas.
        4. The "explanation" must be short, clear, directly related to the sentiment, and not exceed 20 words.
        5. If you **cannot determine the sentiment** or do not understand the text, return:
           {{
               "label": "neutral",
               "confidence": 0.0,
               "explanation": "Unable to determine sentiment"
           }}
    
        Text to analyze:
        {text}
    
        Your response must look EXACTLY like this format, **even if the input is empty or invalid**:
        {{
            "label": "positive|negative|neutral|mixed",
            "confidence": 0.95,
            "explanation": "Brief explanation here, max 20 words"
        }}
        """
        return ChatPromptTemplate.from_template(template)

    async def analyze_sentiment(self, text: str) -> SentimentResult:
        try:
            chain = self.prompt | self.llm | self.parser

            result = await asyncio.wait_for(
                chain.ainvoke({"text": text}),
                timeout=self.request_timeout,
            )

            return SentimentResult.model_validate(result)

        except OutputParserException:
            return SentimentResult.model_validate({
                "label": "neutral",
                "confidence": 0.0,
                "explanation": "Unable to determine sentiment"
            })
        except asyncio.TimeoutError:
            logger.error("Sentiment analysis timeout")
            raise TimeoutError("Sentiment analysis request timed out")
        except ValidationError as exc:
            logger.error(f"Validation error while parsing LLM output: {exc.errors()}")
            raise SentimentAnalysisError(
                message="Invalid response format from LLM",
                details={"validation_errors": exc.errors()},
            )
        except Exception as exc:
            logger.error(f"Sentiment analysis failed: {str(exc)}")
            raise SentimentAnalysisError(
                message="Unexpected error during sentiment analysis",
                details={"error": str(exc)},
            )
