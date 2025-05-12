from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from freqtrade.strategy.interface import merge_informative_pair
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import openai
import os
import json
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import re
import requests
import time
import random

logger = logging.getLogger(__name__)

class LLMStrategy(IStrategy):
    """
    Advanced Strategy using Language Model as a professional trader with tool-calling capabilities
    高级策略：使用语言模型作为具有工具调用能力的专业交易员
    """
    
    INTERFACE_VERSION = 3
    
    # Default ROI - will be overridden by LLM
    # 默认ROI - 将被语言模型覆盖
    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "60": 0.015,
        "120": 0.01
    }

    # Default stoploss - will be overridden by LLM
    # 默认止损 - 将被语言模型覆盖
    stoploss = -0.05
    
    # Trailing stop settings
    # 追踪止损设置
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # Timeframe and informative timeframes
    # 时间周期和信息时间周期
    timeframe = '5m'
    informative_timeframes = ['15m', '1h', '4h', '1d']
    
    # Analysis throttling (in minutes)
    # 分析节流（分钟）
    analysis_interval_minutes = 5
    backtest_analysis_interval = 50  # candles

    # Language model parameters 
    # 语言模型参数
    llm_model = "gpt-4"
    max_tokens = 4096  # Increased to maximum for full freedom
    temperature = 0.4  # Balanced for creativity and consistency
    
    # Memory and state management
    # 内存和状态管理
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 200  # Number of candles needed for analysis
    
    # API Configuration Parameters
    # API配置参数
    api_config = {
        'provider': 'openai',  # 可选: 'openai', 'azure', 'custom'
        'api_url': None,       # 自定义API URL
        'api_key': None,       # API密钥（如果未设置，使用环境变量）
        'retry_attempts': 3,   # 最大重试次数
        'retry_delay': 2,      # 初始重试延迟（秒）
        'timeout': 60,         # API调用超时时间（秒）
    }
    
    def __init__(self, config: dict) -> None:
        """
        初始化策略
        Initialize the strategy
        """
        super().__init__(config)
        self.last_analysis_time = {}
        self.trading_memory = {}  # Stores pair-specific trading history and state
        self.current_analysis = {}  # Stores current analysis results
        self.global_state = {
            'overall_market_sentiment': 'neutral',
            'last_global_analysis': None,
            'model_performance': {
                'successful_trades': 0,
                'total_trades': 0,
                'profit_sum': 0.0,
            },
            'market_regime': {
                'current': 'unknown',
                'last_update': datetime.now() - timedelta(days=1),
                'history': []
            }
        }
        self.scheduled_entries = {}  # Stores scheduled entry points for pairs
        self.scheduled_exits = {}    # Stores scheduled exit points for pairs
        self.risk_manager = {
            'win_rate': 0.5,  # Initial estimated win rate
            'avg_win_pct': 0.0,  # Average win percentage
            'avg_loss_pct': 0.0,  # Average loss percentage
            'expected_value': 0.0,  # Expected value of a trade
            'confidence': 0.0,  # Confidence in the risk estimates
            'max_risk_per_trade': 0.02,  # Maximum risk per trade (2% of account)
            'market_risk_factor': 1.0,  # Current market risk factor (1.0 = normal)
            'last_update': datetime.now()
        }
        
        # 从策略配置中加载API设置
        # Load API settings from strategy config
        if 'llm_api' in config:
            for key in config['llm_api']:
                if key in self.api_config:
                    self.api_config[key] = config['llm_api'][key]
        
        # 设置API密钥
        # Set up API key
        self.setup_api()
        
    def setup_api(self):
        """
        设置API配置和凭证
        Set up API configuration and credentials
        """
        # 设置API密钥，优先使用策略配置中的密钥，其次使用环境变量
        # Set API key, first from strategy config, then from environment variable
        if self.api_config['api_key'] is None:
            if self.api_config['provider'] == 'openai':
                self.api_config['api_key'] = os.environ.get('OPENAI_API_KEY')
            elif self.api_config['provider'] == 'azure':
                self.api_config['api_key'] = os.environ.get('AZURE_OPENAI_API_KEY')
        
        # 设置API URL
        # Set up API URL
        if self.api_config['api_url'] is None:
            if self.api_config['provider'] == 'azure':
                # 针对Azure设置默认API URL格式
                # Default Azure URL format requires endpoint and deployment
                azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
                azure_deployment = os.environ.get('AZURE_OPENAI_DEPLOYMENT')
                if azure_endpoint and azure_deployment:
                    self.api_config['api_url'] = f"{azure_endpoint}/openai/deployments/{azure_deployment}/chat/completions?api-version=2023-05-15"
        
        logger.info(f"API Provider: {self.api_config['provider']}")
        if self.api_config['api_url']:
            logger.info(f"Using custom API URL: {self.api_config['api_url']}")
    
    def call_llm_with_retry(self, messages, tools=None, tool_choice="auto", **kwargs):
        """
        调用语言模型API，带有自动重试机制
        Call the language model API with automatic retry mechanism
        
        Args:
            messages: 消息列表
            tools: 可用工具列表
            tool_choice: 工具选择策略
            **kwargs: 其他参数
            
        Returns:
            API响应对象
        """
        max_attempts = self.api_config['retry_attempts']
        base_delay = self.api_config['retry_delay']
        timeout = self.api_config['timeout']
        
        # 准备基础参数
        # Prepare base parameters
        params = {
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": timeout
        }
        
        # 添加工具相关参数（如果有）
        # Add tool-related parameters if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice
            
        # 添加其他自定义参数
        # Add other custom parameters
        params.update(kwargs)
        
        # 标准化模型名称参数
        # Normalize model name parameter
        if self.api_config['provider'] == 'azure':
            # Azure不使用model参数，而是在URL中指定部署
            # Azure doesn't use model param, it's in the URL
            if 'model' in params:
                del params['model']
        else:
            # 其他提供商需要model参数
            # Other providers need model parameter
            if 'model' not in params:
                params['model'] = self.llm_model
        
        for attempt in range(1, max_attempts + 1):
            try:
                # 根据不同的API提供商选择调用方式
                # Choose API call method based on provider
                if self.api_config['provider'] == 'openai':
                    # 标准OpenAI API调用
                    # Standard OpenAI API call
                    response = openai.ChatCompletion.create(**params)
                    return response
                    
                elif self.api_config['provider'] == 'azure':
                    # Azure OpenAI API调用
                    # Azure OpenAI API call
                    client = openai.AzureOpenAI(
                        api_key=self.api_config['api_key'],
                        api_version="2023-05-15",
                        azure_endpoint=self.api_config['api_url'].split('/openai')[0]
                    )
                    response = client.chat.completions.create(**params)
                    return response
                    
                elif self.api_config['provider'] == 'custom':
                    # 自定义API调用（直接使用requests）
                    # Custom API call (using requests directly)
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_config['api_key']}"
                    }
                    response = requests.post(
                        self.api_config['api_url'],
                        headers=headers,
                        json=params,
                        timeout=timeout
                    )
                    response.raise_for_status()  # 检查HTTP错误
                    return response.json()
                    
                else:
                    raise ValueError(f"Unsupported API provider: {self.api_config['provider']}")
                
            except Exception as e:
                # 如果是最后一次尝试，则重新抛出异常
                # If this is the last attempt, re-raise the exception
                if attempt == max_attempts:
                    logger.error(f"Maximum retry attempts ({max_attempts}) reached. Last error: {str(e)}")
                    raise
                
                # 计算退避延迟时间（指数增长+随机抖动）
                # Calculate backoff delay (exponential + random jitter)
                delay = base_delay * (2 ** (attempt - 1)) * (0.5 + random.random())
                
                logger.warning(f"API call failed (attempt {attempt}/{max_attempts}): {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # 这一行理论上不会被执行，因为在最后一次重试失败时应该已经引发了异常
        # This line should not be executed as we should have raised an exception on the last retry
        raise RuntimeError("Unexpected error in API retry logic")

    def informative_pairs(self):
        """
        定义信息性的货币对/时间周期组合
        Define informative pair/timeframe combinations to use
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        
        for pair in pairs:
            for timeframe in self.informative_timeframes:
                informative_pairs.append((pair, timeframe))
                
        return informative_pairs
        
    def populate_indicators_informative(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算信息性时间周期的指标
        Calculate indicators for informative timeframes
        """
        if not self.dp:
            return dataframe
            
        current_pair = metadata['pair']
        
        # 为信息性时间周期创建空的DataFrame
        # Create empty DataFrame for informative timeframes
        for timeframe in self.informative_timeframes:
            informative = self.dp.get_pair_dataframe(pair=current_pair, timeframe=timeframe)
            
            # 在信息性时间周期上计算指标
            # Calculate indicators on informative timeframe
            informative['rsi'] = ta.RSI(informative, timeperiod=14)
            informative['ema_9'] = ta.EMA(informative, timeperiod=9)
            informative['ema_21'] = ta.EMA(informative, timeperiod=21)
            informative['ema_50'] = ta.EMA(informative, timeperiod=50)
            informative['ema_200'] = ta.EMA(informative, timeperiod=200)
            
            # MACD指标
            # MACD indicator
            macd = ta.MACD(informative)
            informative['macd'] = macd['macd']
            informative['macdsignal'] = macd['macdsignal']
            informative['macdhist'] = macd['macdhist']
            
            # 布林带指标
            # Bollinger Bands indicator
            bollinger = ta.BBANDS(informative, timeperiod=20)
            informative['bb_lowerband'] = bollinger['lowerband']
            informative['bb_middleband'] = bollinger['middleband']
            informative['bb_upperband'] = bollinger['upperband']
            informative['bb_width'] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']
            
            # ATR指标
            # ATR indicator
            informative['atr'] = ta.ATR(informative, timeperiod=14)
            
            # 将结果与原始dataframe合并，使用时间周期作为后缀
            # Merge with original dataframe with suffix of timeframe
            suffix = f"_{timeframe}"
            dataframe = merge_informative_pair(dataframe, informative, self.timeframe, timeframe, suffix=suffix)
            
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        计算技术指标，但不将所有指标暴露给LLM以减少token使用
        工具将根据需要访问这些指标
        
        Calculate technical indicators but don't expose all to LLM
        to reduce token usage - tools will access these as needed
        """
        # 首先获取信息性时间周期的指标
        # First get informative timeframe indicators
        if self.dp and len(self.informative_timeframes) > 0:
            dataframe = self.populate_indicators_informative(dataframe, metadata)
            
        # 基本价格信息
        # Basic price info
        dataframe['hlc3'] = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
        
        # 成交量指标
        # Volume indicators
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_ma']
        
        # 趋势指标
        # Trend indicators
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=21)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=7)
        
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)
        
        # 波动率指标
        # Volatility indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['natr'] = dataframe['atr'] / dataframe['close'] * 100  # Normalized ATR as percentage
        
        # MACD指标
        # MACD indicator
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # 布林带指标
        # Bollinger Bands indicator
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_width'] = (bollinger['upperband'] - bollinger['lowerband']) / bollinger['middleband']
        
        # 随机指标
        # Stochastic indicator
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        
        # ADX指标
        # ADX indicator
        dataframe['adx'] = ta.ADX(dataframe)
        
        # 支撑和阻力水平
        # Support and resistance levels
        for period in [14, 30, 90]:
            dataframe[f'resistance_{period}'] = dataframe['high'].rolling(period).max()
            dataframe[f'support_{period}'] = dataframe['low'].rolling(period).min()
        
        # 价格动量
        # Price momentum
        for period in [1, 6, 12, 24]:
            dataframe[f'return_{period}'] = dataframe['close'].pct_change(period) * 100
        
        return dataframe

    def should_analyze_market(self, pair: str, dataframe: DataFrame) -> bool:
        """
        根据以下因素确定是否应该进行新的市场分析：
        1. 来自先前模型分析的计划检查
        2. 上次分析后经过的时间（生产环境）
        3. 上次分析后的K线数量（回测环境）
        
        Determine if we should perform a new market analysis based on:
        1. Scheduled checks from previous model analysis
        2. Time since last analysis (production)
        3. Number of candles since last analysis (backtesting)
        """
        # 检查是否有针对这个货币对的计划检查
        # Check if we have a scheduled check for this pair
        if self.process_scheduled_checks(dataframe, {'pair': pair}):
            return True
            
        # 对于回测，检查K线计数
        # For backtesting, check candle count
        if self.dp and self.dp.runmode.value == 'backtest':
            pair_state = self.trading_memory.get(pair, {})
            last_analyzed_candle = pair_state.get('last_analyzed_candle', 0)
            current_candle = len(dataframe) - 1
            
            if current_candle - last_analyzed_candle >= self.backtest_analysis_interval:
                if pair in self.trading_memory:
                    self.trading_memory[pair]['last_analyzed_candle'] = current_candle
                else:
                    self.trading_memory[pair] = {'last_analyzed_candle': current_candle}
                return True
            return False
            
        # 对于生产环境，检查时间间隔
        # For production, check time interval
        current_time = datetime.now()
        last_time = self.last_analysis_time.get(pair)
        
        # 检查是否有该货币对的未平仓头寸
        # Check if we have a position open for this pair
        has_open_position = False
        if self.config.get('trading_mode', 'spot') != 'test':
            try:
                from freqtrade.persistence.models import Trade
                for trade in Trade.get_trades_proxy(is_open=True):
                    if trade.pair == pair:
                        has_open_position = True
                        break
            except Exception as e:
                logger.error(f"Error checking open positions: {e}")

        # 如果没有计划检查并且没有未平仓头寸，使用常规时间间隔检查
        # If we have no scheduled checks and no open position, use regular time interval check
        if not has_open_position and (not last_time or (current_time - last_time) > timedelta(minutes=self.analysis_interval_minutes)):
            self.last_analysis_time[pair] = current_time
            return True
            
        return False
    
    def generate_tool_schema(self) -> List[Dict]:
        """
        Generate the schema for tools the LLM can call
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_price_data",
                    "description": "Get price data for a specific timeframe",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lookback_periods": {
                                "type": "integer",
                                "description": "Number of candles to look back"
                            },
                            "data_type": {
                                "type": "string",
                                "enum": ["close", "open", "high", "low", "volume", "hlc3"],
                                "description": "Type of price data to retrieve"
                            }
                        },
                        "required": ["lookback_periods", "data_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_multi_timeframe_data",
                    "description": "Get data from higher timeframes for multi-timeframe analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timeframe": {
                                "type": "string",
                                "enum": ["15m", "1h", "4h", "1d"],
                                "description": "Larger timeframe to analyze"
                            },
                            "indicator": {
                                "type": "string",
                                "enum": ["rsi", "macd", "macdsignal", "macdhist", "ema_9", "ema_21", "ema_50", "ema_200", "bb_lowerband", "bb_middleband", "bb_upperband", "bb_width", "atr"],
                                "description": "Indicator to retrieve from the larger timeframe"
                            },
                            "lookback_periods": {
                                "type": "integer",
                                "description": "Number of larger timeframe candles to look back"
                            }
                        },
                        "required": ["timeframe", "indicator"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_technical_indicator",
                    "description": "Get technical indicator values",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "indicator": {
                                "type": "string",
                                "enum": ["rsi", "rsi_slow", "rsi_fast", "macd", "macdsignal", "macdhist", "bb_width", "atr", "natr", "ema_9", "ema_21", "ema_50", "ema_200", "sma_50", "sma_200", "slowk", "slowd", "adx"],
                                "description": "Technical indicator to retrieve"
                            },
                            "lookback_periods": {
                                "type": "integer",
                                "description": "Number of periods to look back"
                            }
                        },
                        "required": ["indicator", "lookback_periods"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_price_momentum",
                    "description": "Get price momentum/return over specific periods",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "periods": {
                                "type": "integer",
                                "enum": [1, 6, 12, 24],
                                "description": "Number of periods to calculate momentum for"
                            }
                        },
                        "required": ["periods"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_support_resistance",
                    "description": "Get support and resistance levels",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "period": {
                                "type": "integer",
                                "enum": [14, 30, 90],
                                "description": "Period for support/resistance calculation"
                            }
                        },
                        "required": ["period"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_structure",
                    "description": "Get overall market structure assessment",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_trade_history",
                    "description": "Get history of previous trades for this pair",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_advanced_analysis",
                    "description": "Perform advanced market analysis",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "enum": ["correlation", "divergence", "volatility_regime", "fractal_patterns"],
                                "description": "Type of advanced analysis to perform"
                            },
                            "lookback_periods": {
                                "type": "integer",
                                "description": "Number of periods to analyze"
                            }
                        },
                        "required": ["analysis_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_risk_management_info",
                    "description": "Get current risk management parameters and trade statistics",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_regime",
                    "description": "Get information about the current market regime and conditions",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        return tools

    def handle_tool_calls(self, tool_calls: List[Dict], dataframe: DataFrame, metadata: dict) -> List[Dict]:
        """
        Process tool calls from the LLM and return results
        """
        results = []
        pair = metadata['pair']
        
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            
            try:
                if function_name == "get_price_data":
                    lookback = function_args["lookback_periods"]  # No more cap on lookback periods
                    data_type = function_args["data_type"]
                    
                    # Get the most recent data points
                    latest_values = dataframe[data_type].iloc[-lookback:].tolist()
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "values": latest_values,
                            "current": latest_values[-1],
                            "min": min(latest_values),
                            "max": max(latest_values),
                            "avg": sum(latest_values) / len(latest_values)
                        })
                    })
                
                elif function_name == "get_multi_timeframe_data":
                    timeframe = function_args["timeframe"]
                    indicator = function_args["indicator"]
                    lookback = function_args["lookback_periods"]
                    
                    # Get data from higher timeframe
                    higher_timeframe_data = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
                    higher_values = higher_timeframe_data[indicator].iloc[-lookback:].tolist()
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "values": higher_values,
                            "current": higher_values[-1],
                            "min": min(higher_values),
                            "max": max(higher_values),
                            "avg": sum(higher_values) / len(higher_values)
                        })
                    })
                
                elif function_name == "get_technical_indicator":
                    indicator = function_args["indicator"]
                    lookback = function_args["lookback_periods"]  # No more cap on lookback periods
                    
                    latest_values = dataframe[indicator].iloc[-lookback:].tolist()
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "values": latest_values,
                            "current": latest_values[-1], 
                            "previous": latest_values[-2] if len(latest_values) > 1 else None,
                            "min": min(latest_values),
                            "max": max(latest_values),
                            "avg": sum(latest_values) / len(latest_values)
                        })
                    })
                
                elif function_name == "get_price_momentum":
                    periods = function_args["periods"]
                    momentum = dataframe[f'return_{periods}'].iloc[-1]
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "momentum": momentum,
                            "is_positive": momentum > 0,
                            "magnitude": abs(momentum)
                        })
                    })
                
                elif function_name == "get_support_resistance":
                    period = function_args["period"]
                    current_price = dataframe['close'].iloc[-1]
                    resistance = dataframe[f'resistance_{period}'].iloc[-1]
                    support = dataframe[f'support_{period}'].iloc[-1]
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "support": support,
                            "resistance": resistance,
                            "distance_to_support": ((current_price - support) / current_price) * 100,
                            "distance_to_resistance": ((resistance - current_price) / current_price) * 100
                        })
                    })
                
                elif function_name == "get_market_structure":
                    # Calculate market structure
                    current_price = dataframe['close'].iloc[-1]
                    ema_9 = dataframe['ema_9'].iloc[-1]
                    ema_21 = dataframe['ema_21'].iloc[-1]
                    ema_50 = dataframe['ema_50'].iloc[-1]
                    ema_200 = dataframe['ema_200'].iloc[-1]
                    
                    sma_50 = dataframe['sma_50'].iloc[-1]
                    sma_200 = dataframe['sma_200'].iloc[-1]
                    
                    # Determine trend
                    if ema_9 > ema_21 > ema_50 > ema_200:
                        trend = "Strong Uptrend"
                    elif ema_9 > ema_21 > ema_50 and ema_50 < ema_200:
                        trend = "Weak Uptrend"
                    elif ema_9 < ema_21 < ema_50 < ema_200:
                        trend = "Strong Downtrend"
                    elif ema_9 < ema_21 < ema_50 and ema_50 > ema_200:
                        trend = "Weak Downtrend"
                    else:
                        trend = "Ranging"
                    
                    # Golden/Death cross check
                    golden_cross = sma_50 > sma_200 and dataframe['sma_50'].iloc[-2] <= dataframe['sma_200'].iloc[-2]
                    death_cross = sma_50 < sma_200 and dataframe['sma_50'].iloc[-2] >= dataframe['sma_200'].iloc[-2]
                    
                    volatility = dataframe['natr'].iloc[-1]
                    avg_volatility = dataframe['natr'].iloc[-20:].mean()
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "trend": trend,
                            "golden_cross": golden_cross,
                            "death_cross": death_cross,
                            "above_ema_200": current_price > ema_200,
                            "above_ema_50": current_price > ema_50,
                            "current_volatility": volatility,
                            "is_high_volatility": volatility > avg_volatility * 1.5,
                            "is_low_volatility": volatility < avg_volatility * 0.5
                        })
                    })
                
                elif function_name == "get_trade_history":
                    # Get complete trading history for this pair
                    trade_history = self.trading_memory.get(pair, {}).get('trade_history', [])
                    last_trade = self.trading_memory.get(pair, {}).get('last_trade', {})
                    winning_trades = self.trading_memory.get(pair, {}).get('winning_trades', 0)
                    total_trades = self.trading_memory.get(pair, {}).get('total_trades', 0)
                    cumulative_profit = self.trading_memory.get(pair, {}).get('cumulative_profit', 0.0)
                    
                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "trade_count": len(trade_history),
                            "win_rate": win_rate,
                            "cumulative_profit": cumulative_profit,
                            "last_trade": last_trade,
                            "all_trades": trade_history  # Return the full history
                        })
                    })
                
                elif function_name == "perform_advanced_analysis":
                    # Advanced market analysis capability
                    analysis_type = function_args.get("analysis_type", "correlation")
                    lookback = function_args.get("lookback_periods", 100)
                    
                    result = {}
                    
                    if analysis_type == "correlation":
                        # Calculate correlation between indicators
                        corr = {}
                        indicators = ["rsi", "macd", "bb_width", "atr", "adx"]
                        for i1 in indicators:
                            for i2 in indicators:
                                if i1 != i2:
                                    corr[f"{i1}_vs_{i2}"] = dataframe[i1].iloc[-lookback:].corr(dataframe[i2].iloc[-lookback:])
                        result["correlations"] = corr
                        
                    elif analysis_type == "divergence":
                        # Check for divergences
                        price_action = dataframe['close'].iloc[-lookback:].values
                        rsi_action = dataframe['rsi'].iloc[-lookback:].values
                        
                        # Check if price made higher high but RSI made lower high (bearish divergence)
                        if (len(price_action) > 20):
                            price_trend = np.polyfit(range(len(price_action)), price_action, 1)[0]
                            rsi_trend = np.polyfit(range(len(rsi_action)), rsi_action, 1)[0]
                            
                            result["price_trend"] = price_trend
                            result["rsi_trend"] = rsi_trend
                            result["bearish_divergence"] = price_trend > 0 and rsi_trend < 0
                            result["bullish_divergence"] = price_trend < 0 and rsi_trend > 0
                    
                    elif analysis_type == "volatility_regime":
                        # Detect volatility regime
                        current_volatility = dataframe['natr'].iloc[-1]
                        historical_volatility = dataframe['natr'].iloc[-lookback:-1].mean()
                        volatility_percentile = sum(dataframe['natr'].iloc[-lookback:-1] < current_volatility) / (lookback-1)
                        
                        result["current_volatility"] = current_volatility
                        result["historical_volatility"] = historical_volatility
                        result["volatility_percentile"] = volatility_percentile
                        result["volatility_regime"] = "high" if volatility_percentile > 0.8 else "low" if volatility_percentile < 0.2 else "normal"
                        
                    elif analysis_type == "fractal_patterns":
                        # Detect price fractals
                        highs = []
                        lows = []
                        
                        # Basic fractal detection
                        for i in range(2, min(lookback, len(dataframe)-2)):
                            idx = -lookback + i
                            # High fractal
                            if (dataframe['high'].iloc[idx] > dataframe['high'].iloc[idx-1] and 
                                dataframe['high'].iloc[idx] > dataframe['high'].iloc[idx-2] and
                                dataframe['high'].iloc[idx] > dataframe['high'].iloc[idx+1] and
                                dataframe['high'].iloc[idx] > dataframe['high'].iloc[idx+2]):
                                highs.append({"price": dataframe['high'].iloc[idx], "position": i})
                            
                            # Low fractal
                            if (dataframe['low'].iloc[idx] < dataframe['low'].iloc[idx-1] and 
                                dataframe['low'].iloc[idx] < dataframe['low'].iloc[idx-2] and
                                dataframe['low'].iloc[idx] < dataframe['low'].iloc[idx+1] and
                                dataframe['low'].iloc[idx] < dataframe['low'].iloc[idx+2]):
                                lows.append({"price": dataframe['low'].iloc[idx], "position": i})
                        
                        result["fractal_highs"] = highs[-5:] if len(highs) > 5 else highs
                        result["fractal_lows"] = lows[-5:] if len(lows) > 5 else lows
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps(result)
                    })
                
                elif function_name == "get_risk_management_info":
                    # Get risk management parameters
                    risk_params = self.update_risk_parameters()
                    
                    # Get global performance metrics
                    perf = self.global_state['model_performance']
                    
                    # Get pair-specific metrics if available
                    pair_stats = {}
                    if pair in self.trading_memory:
                        pair_stats = {
                            'total_trades': self.trading_memory[pair].get('total_trades', 0),
                            'winning_trades': self.trading_memory[pair].get('winning_trades', 0),
                            'win_rate': (self.trading_memory[pair].get('winning_trades', 0) / 
                                       self.trading_memory[pair].get('total_trades', 1) 
                                       if self.trading_memory[pair].get('total_trades', 0) > 0 else 0),
                            'cumulative_profit': self.trading_memory[pair].get('cumulative_profit', 0.0)
                        }
                    
                    # Calculate position sizing recommendations for different scenarios
                    position_recommendations = []
                    current_price = dataframe['close'].iloc[-1]
                    
                    for stop_pct in [2, 5, 10]:
                        stop_price = current_price * (1 - stop_pct/100)
                        position_size = self.calculate_position_size(
                            entry_price=current_price,
                            stop_loss=stop_price,
                            confidence=0.8  # Default high confidence
                        )
                        position_recommendations.append({
                            'stop_loss_pct': stop_pct,
                            'recommended_size': position_size
                        })
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "risk_parameters": {
                                "win_rate": risk_params['win_rate'],
                                "avg_win_pct": risk_params['avg_win_pct'],
                                "avg_loss_pct": risk_params['avg_loss_pct'],
                                "expected_value": risk_params['expected_value'],
                                "market_risk_factor": risk_params['market_risk_factor']
                            },
                            "global_performance": {
                                "total_trades": perf['total_trades'],
                                "successful_trades": perf['successful_trades'],
                                "win_rate": perf['successful_trades'] / perf['total_trades'] if perf['total_trades'] > 0 else 0,
                                "total_profit": perf['profit_sum']
                            },
                            "pair_statistics": pair_stats,
                            "position_sizing_recommendations": position_recommendations
                        })
                    })
                
                elif function_name == "get_market_regime":
                    # Ensure the market regime is up to date
                    current_regime = self.detect_market_regime(dataframe, pair)
                    regime_data = self.global_state['market_regime']
                    
                    # Get additional market context
                    try:
                        # Get market data from daily timeframe
                        market_data = self.dp.get_pair_dataframe(pair=pair, timeframe='1d')
                        
                        # Calculate key metrics
                        if len(market_data) >= 14:
                            # RSI
                            market_data['rsi'] = ta.RSI(market_data, timeperiod=14)
                            current_rsi = market_data['rsi'].iloc[-1]
                            
                            # Trend strength (using ADX)
                            market_data['adx'] = ta.ADX(market_data)
                            current_adx = market_data['adx'].iloc[-1]
                            
                            # Distance from 200 MA
                            market_data['sma_200'] = ta.SMA(market_data['close'], timeperiod=200)
                            if not np.isnan(market_data['sma_200'].iloc[-1]):
                                distance_200ma = (market_data['close'].iloc[-1] / market_data['sma_200'].iloc[-1] - 1) * 100
                            else:
                                distance_200ma = 0
                                
                            # Recent performance
                            week_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-7] - 1) * 100 if len(market_data) >= 7 else 0
                            month_change = (market_data['close'].iloc[-1] / market_data['close'].iloc[-30] - 1) * 100 if len(market_data) >= 30 else 0
                            
                            additional_context = {
                                'current_rsi': current_rsi,
                                'trend_strength_adx': current_adx,
                                'distance_from_200ma_pct': distance_200ma,
                                'week_change_pct': week_change,
                                'month_change_pct': month_change
                            }
                        else:
                            additional_context = {}
                    except Exception as e:
                        logger.error(f"Error calculating additional market context: {str(e)}")
                        additional_context = {}
                    
                    # Get recommendations based on regime
                    recommendations = []
                    
                    if "bullish" in current_regime:
                        if "high_volatility" in current_regime:
                            recommendations = [
                                "Consider using wider stop losses due to high volatility",
                                "Look for pullbacks to key support levels for entries",
                                "Use trailing stops to protect profits in volatile uptrend"
                            ]
                        else:
                            recommendations = [
                                "Strong uptrend with normal volatility - favor bullish setups",
                                "Consider longer holding periods in this regime",
                                "Focus on breakouts above resistance levels"
                            ]
                    elif "bearish" in current_regime:
                        if "high_volatility" in current_regime:
                            recommendations = [
                                "High risk environment - consider smaller position sizes",
                                "Look for bounces to resistance for short entries",
                                "Be cautious of sudden reversal rallies"
                            ]
                        else:
                            recommendations = [
                                "Established downtrend - avoid fighting the trend",
                                "Focus on capital preservation",
                                "Look for oversold bounces for quick profits"
                            ]
                    else:  # sideways
                        if "high_volatility" in current_regime:
                            recommendations = [
                                "Range-bound with high volatility - focus on range extremes",
                                "Use oscillators like RSI for overbought/oversold signals",
                                "Reduce position sizes in choppy conditions"
                            ]
                        else:
                            recommendations = [
                                "Low volatility consolidation - prepare for breakout",
                                "Watch for volume increase as sign of emerging trend",
                                "Consider waiting for clearer directional signals"
                            ]
                    
                    results.append({
                        "tool_call_id": tool_call["id"],
                        "output": json.dumps({
                            "current_regime": current_regime,
                            "regime_history": regime_data['history'],
                            "last_regime_change": regime_data['history'][-1] if regime_data['history'] else None,
                            "market_context": additional_context,
                            "trading_recommendations": recommendations
                        })
                    })
                
            except Exception as e:
                results.append({
                    "tool_call_id": tool_call["id"],
                    "output": json.dumps({"error": str(e)})
                })
                
        return results

    def create_trading_decision(self, dataframe: DataFrame, metadata: dict) -> Dict:
        """
        Let the LLM analyze the market by calling tools and making a trading decision
        """
        pair = metadata['pair']
        current_price = dataframe['close'].iloc[-1]
        current_time = datetime.now()
        
        # Check if we already have an open position
        open_position = False
        if self.config.get('trading_mode', 'spot') != 'test':
            try:
                from freqtrade.persistence.models import Trade
                for trade in Trade.get_trades_proxy(is_open=True):
                    if trade.pair == pair:
                        open_position = True
                        break
            except Exception as e:
                logger.error(f"Error checking open positions: {e}")
                
        # Initialize trading memory for this pair if it doesn't exist
        if pair not in self.trading_memory:
            self.trading_memory[pair] = {
                'trade_history': [],
                'last_trade': None,
                'cumulative_profit': 0.0,
                'winning_trades': 0,
                'total_trades': 0
            }
        
        # System prompt for the trader LLM
        system_prompt = """You are an expert cryptocurrency trader with full freedom to analyze and make decisions.
        
        Guidelines:
        1. Use the available tools to gather any information you need for your analysis
        2. You have complete freedom to develop your own trading strategy and approach
        3. You can request as much data as you need and perform in-depth analysis
        4. You can consider any factors you deem relevant for your decision
        5. You can form your own theories about market movement and test them with data
        6. Balance risk and reward based on your own judgment
        7. You can decide WHEN to next analyze this pair rather than checking every candle
        
        Your only constraint is to provide your final recommendation in a specific format.
        """
        
        # User prompt with basic market context and position information
        position_context = "You currently have an OPEN POSITION." if open_position else "You currently have NO OPEN POSITION."
        
        # Add scheduled entry information if available
        next_scheduled = ""
        if pair in self.scheduled_entries:
            next_scheduled = f"You previously scheduled to check this pair again when price reaches {self.scheduled_entries[pair]['price']} or at {self.scheduled_entries[pair]['time']}."
        
        user_prompt = f"""Analyze the trading pair {pair} at the current price of {current_price}.
        
        IMPORTANT - POSITION STATUS: {position_context}
        {next_scheduled}
        
        You have complete freedom to use the tools to gather whatever information you need. Take your time to thoroughly analyze the market before making a decision.
        
        You can:
        - Look at as much historical price data as you want
        - Analyze any technical indicators you find useful
        - Examine support/resistance levels
        - Study the overall market structure
        - Review previous trades for this pair
        
        After your analysis, make a recommendation using the exact format below:
        
        RECOMMENDATION: [BUY/SELL/HOLD]
        CONFIDENCE: [0.0-1.0] 
        ENTRY_PRICE: [specific price or "MARKET"]
        STOP_LOSS: [specific price or percentage]
        TAKE_PROFIT: [specific price or percentage]
        POSITION_SIZE: [0.0-1.0]
        REASONING: [detailed explanation of your analysis and decision]
        MARKET_SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
        NEXT_CHECK: [PRICE/DATETIME/INDICATOR/CANDLES]
        NEXT_CHECK_CONDITION: [e.g. "When price reaches 45000" or "After 24 candles" or "When RSI crosses below 30"]
        """
        
        try:
            # Call the OpenAI API with tool-calling capability
            response = self.call_llm_with_retry(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=self.generate_tool_schema(),
                tool_choice="auto"
            )
            
            # Process the response (which may include tool calls)
            decision = self.process_llm_response(response, dataframe, metadata)
            return decision
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return {
                "action": "hold",
                "confidence": 0,
                "reasoning": f"Error in analysis: {str(e)}"
            }

    def process_llm_response(self, response, dataframe: DataFrame, metadata: dict) -> Dict:
        """
        Process the LLM response, handling any tool calls iteratively until a final decision is made
        """
        messages = [
            {"role": "system", "content": "You are an expert cryptocurrency trader."},
            {"role": "user", "content": f"Analyze the trading pair {metadata['pair']} and make a decision."}
        ]
        
        # Keep track of all messages in the conversation
        messages.append({
            "role": response.choices[0].message.role,
            "content": response.choices[0].message.content
        })
        
        # Handle tool calls iteratively until we get a final decision
        max_iterations = 10  # Increased from 5 to give model more thinking time
        iteration = 0
        
        while iteration < max_iterations:
            # If no tool calls, parse the final response
            if "tool_calls" not in response.choices[0].message:
                decision = self.parse_trading_decision(response.choices[0].message.content)
                
                # Process scheduling information if available
                if decision.get('next_check_type') and decision.get('next_check_condition'):
                    self.schedule_next_check(metadata['pair'], decision, dataframe)
                    
                return decision
                
            # Process tool calls
            tool_calls = response.choices[0].message.tool_calls
            tool_results = self.handle_tool_calls(tool_calls, dataframe, metadata)
            
            # Add the tool responses to the conversation
            messages.append({
                "role": "tool",
                "tool_calls": tool_calls,
                "tool_call_results": tool_results
            })
            
            # Make another API call with the updated conversation
            try:
                response = self.call_llm_with_retry(
                    messages=messages,
                    tools=self.generate_tool_schema(),
                    tool_choice="auto"
                )
                
                # Add the new response to the conversation
                messages.append({
                    "role": response.choices[0].message.role,
                    "content": response.choices[0].message.content
                })
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in tool processing: {str(e)}")
                return {
                    "action": "hold",
                    "confidence": 0,
                    "reasoning": f"Error in tool processing: {str(e)}"
                }
        
        # If we've reached max iterations without a decision, return a hold
        return {
            "action": "hold",
            "confidence": 0,
            "reasoning": "Could not reach a decision after maximum iterations"
        }

    def schedule_next_check(self, pair: str, decision: Dict, dataframe: DataFrame):
        """
        Schedule the next time to check this pair based on the model's decision
        """
        next_check_type = decision.get('next_check_type')
        next_check_condition = decision.get('next_check_condition', '')
        
        if not next_check_type or not next_check_condition:
            return
            
        schedule_info = {
            'type': next_check_type,
            'condition': next_check_condition,
            'created_at': datetime.now()
        }
        
        try:
            # Parse condition based on type
            if next_check_type == 'PRICE':
                # Try to extract a price value from the condition
                price_match = re.search(r'(\d+\.?\d*)', next_check_condition)
                if price_match:
                    schedule_info['value'] = price_match.group(1)
                    logger.info(f"Scheduled next check for {pair} when price reaches {schedule_info['value']}")
                
            elif next_check_type == 'DATETIME':
                # Try to extract a datetime or relative time
                if 'hours' in next_check_condition.lower():
                    hours_match = re.search(r'(\d+)\s*hours', next_check_condition.lower())
                    if hours_match:
                        hours = int(hours_match.group(1))
                        target_time = datetime.now() + timedelta(hours=hours)
                        schedule_info['value'] = target_time.isoformat()
                elif 'minutes' in next_check_condition.lower():
                    minutes_match = re.search(r'(\d+)\s*minutes', next_check_condition.lower())
                    if minutes_match:
                        minutes = int(minutes_match.group(1))
                        target_time = datetime.now() + timedelta(minutes=minutes)
                        schedule_info['value'] = target_time.isoformat()
                logger.info(f"Scheduled next check for {pair} at {schedule_info['value']}")
                
            elif next_check_type == 'CANDLES':
                # Try to extract number of candles to wait
                candles_match = re.search(r'(\d+)\s*candles', next_check_condition.lower())
                if candles_match:
                    schedule_info['value'] = candles_match.group(1)
                    schedule_info['candles_waited'] = 0
                    logger.info(f"Scheduled next check for {pair} after {schedule_info['value']} candles")
                
            elif next_check_type == 'INDICATOR':
                # Try to extract indicator, operator and value
                indicator_match = re.search(r'([a-zA-Z0-9_]+)\s*(>|<|=|crosses\s*above|crosses\s*below|reaches)\s*(\d+\.?\d*)', next_check_condition.lower())
                if indicator_match:
                    indicator = indicator_match.group(1)
                    operator_text = indicator_match.group(2)
                    value = indicator_match.group(3)
                    
                    # Map text operators to symbols
                    operator = '>'
                    if '<' in operator_text or 'below' in operator_text:
                        operator = '<'
                    elif '=' in operator_text or 'reaches' in operator_text:
                        operator = '='
                    
                    schedule_info['indicator'] = indicator
                    schedule_info['operator'] = operator
                    schedule_info['value'] = value
                    logger.info(f"Scheduled next check for {pair} when {indicator} {operator} {value}")
            
            # Store the scheduling information
            self.scheduled_entries[pair] = schedule_info
            
        except Exception as e:
            logger.error(f"Error scheduling next check for {pair}: {str(e)}")
            # Still store basic scheduling info
            self.scheduled_entries[pair] = schedule_info

    def parse_trading_decision(self, content: str) -> Dict:
        """
        Parse the LLM's final trading decision from its text response
        """
        decision = {
            "action": "hold",
            "confidence": 0,
            "entry_price": "MARKET",
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0.5,
            "reasoning": "",
            "market_sentiment": "NEUTRAL",
            "next_check_type": None,
            "next_check_condition": None
        }
        
        try:
            # Extract recommendation
            if "RECOMMENDATION: BUY" in content:
                decision["action"] = "buy"
            elif "RECOMMENDATION: SELL" in content:
                decision["action"] = "sell"
                
            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE: (0\.\d+|1\.0)", content)
            if confidence_match:
                decision["confidence"] = float(confidence_match.group(1))
                
            # Extract entry price
            entry_match = re.search(r"ENTRY_PRICE: (MARKET|\d+\.?\d*)", content)
            if entry_match:
                entry = entry_match.group(1)
                decision["entry_price"] = entry if entry == "MARKET" else float(entry)
                
            # Extract stop loss
            stop_match = re.search(r"STOP_LOSS: (-?\d+\.?\d*%|-?\d+\.?\d*)", content)
            if stop_match:
                stop_value = stop_match.group(1)
                decision["stop_loss"] = float(stop_value.replace("%", "")) / 100 if "%" in stop_value else float(stop_value)
                
            # Extract take profit
            tp_match = re.search(r"TAKE_PROFIT: (\d+\.?\d*%|\d+\.?\d*)", content)
            if tp_match:
                tp_value = tp_match.group(1)
                decision["take_profit"] = float(tp_value.replace("%", "")) / 100 if "%" in tp_value else float(tp_value)
                
            # Extract position size
            size_match = re.search(r"POSITION_SIZE: (0\.\d+|1\.0)", content)
            if size_match:
                decision["position_size"] = float(size_match.group(1))
                
            # Extract reasoning - capture everything between REASONING: and MARKET_SENTIMENT:
            reason_match = re.search(r"REASONING: (.*?)(?=MARKET_SENTIMENT:|$)", content, re.DOTALL)
            if reason_match:
                decision["reasoning"] = reason_match.group(1).strip()
                
            # Extract market sentiment
            sentiment_match = re.search(r"MARKET_SENTIMENT: (BULLISH|BEARISH|NEUTRAL)", content)
            if sentiment_match:
                decision["market_sentiment"] = sentiment_match.group(1)
                
            # Extract next check type
            next_check_match = re.search(r"NEXT_CHECK: (PRICE|DATETIME|INDICATOR|CANDLES|[A-Za-z0-9_]+)", content)
            if next_check_match:
                decision["next_check_type"] = next_check_match.group(1).upper()
                
            # Extract next check condition
            next_check_condition_match = re.search(r"NEXT_CHECK_CONDITION: (.*?)(?=$|\n\n)", content, re.DOTALL)
            if next_check_condition_match:
                decision["next_check_condition"] = next_check_condition_match.group(1).strip()
                
        except Exception as e:
            logger.error(f"Error parsing trading decision: {str(e)}")
            decision["reasoning"] = f"Error parsing response: {str(e)}"
            
        return decision

    def update_trade_history(self, pair: str, decision: Dict, is_entry: bool, price: float):
        """
        Update the pair's trading history with the current decision
        """
        if pair not in self.trading_memory:
            self.trading_memory[pair] = {
                'trade_history': [],
                'last_trade': None,
                'cumulative_profit': 0.0,
                'winning_trades': 0,
                'total_trades': 0
            }
            
        # If this is a trade exit, calculate profit/loss
        if not is_entry and self.trading_memory[pair].get('last_trade') and self.trading_memory[pair]['last_trade'].get('is_entry'):
            last_entry = self.trading_memory[pair]['last_trade']
            entry_price = last_entry.get('price', 0)
            
            if entry_price > 0:
                profit_pct = ((price - entry_price) / entry_price) * 100 if last_entry.get('direction') == 'long' else ((entry_price - price) / entry_price) * 100
                
                # Record profit/loss
                trade_record = {
                    'entry_time': last_entry.get('time'),
                    'exit_time': datetime.now(),
                    'entry_price': entry_price,
                    'exit_price': price,
                    'direction': last_entry.get('direction'),
                    'profit_pct': profit_pct,
                    'entry_reasoning': last_entry.get('reasoning', ''),
                    'exit_reasoning': decision.get('reasoning', '')
                }
                
                self.trading_memory[pair]['trade_history'].append(trade_record)
                self.trading_memory[pair]['cumulative_profit'] += profit_pct
                self.trading_memory[pair]['total_trades'] += 1
                
                # Update global performance metrics
                self.global_state['model_performance']['total_trades'] += 1
                self.global_state['model_performance']['profit_sum'] += profit_pct
                
                if profit_pct > 0:
                    self.trading_memory[pair]['winning_trades'] += 1
                    self.global_state['model_performance']['successful_trades'] += 1
                
                # Update risk management system with trade result
                self.update_risk_parameters(trade_record)
                
                logger.info(f"Trade closed for {pair}: {profit_pct:.2f}% profit")
        
        # Record the current trade
        entry_price = price
        stop_loss_price = None
        
        # Calculate stop loss price if provided as percentage
        if decision.get('stop_loss') and isinstance(decision['stop_loss'], float):
            if decision['stop_loss'] < 0:  # It's a percentage
                stop_loss_price = price * (1 + decision['stop_loss'])
            else:  # It's an absolute price
                stop_loss_price = decision['stop_loss']
        
        # Calculate position size using risk management
        position_size = decision.get('position_size', 0.5)
        if is_entry and decision.get('action') == 'buy':
            # Use the adaptive position sizing if we have stop loss
            if stop_loss_price:
                position_size = self.calculate_position_size(
                    entry_price=price,
                    stop_loss=stop_loss_price,
                    confidence=decision.get('confidence', 0.7)
                )
        
        self.trading_memory[pair]['last_trade'] = {
            'time': datetime.now(),
            'price': price,
            'is_entry': is_entry,
            'direction': 'long' if decision.get('action') == 'buy' else 'short',
            'stop_loss': decision.get('stop_loss'),
            'take_profit': decision.get('take_profit'),
            'confidence': decision.get('confidence'),
            'reasoning': decision.get('reasoning'),
            'position_size': position_size
        }

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate entry signals based on LLM analysis
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_tag'] = ''
        
        # Check if we should analyze this pair at this time
        if not self.should_analyze_market(metadata['pair'], dataframe):
            return dataframe
            
        # Get LLM trading decision
        decision = self.create_trading_decision(dataframe, metadata)
        
        # Store the decision for this pair
        self.current_analysis[metadata['pair']] = decision
        
        # Generate entry signal if the LLM recommends a buy with sufficient confidence
        last_index = len(dataframe) - 1
        
        if decision['action'] == 'buy' and decision['confidence'] >= 0.7:
            dataframe.loc[last_index, 'enter_long'] = 1
            
            # Set custom tag with decision details
            conf = int(decision['confidence'] * 100)
            reason_brief = decision.get('reasoning', '')[:20].replace(' ', '_')
            dataframe.loc[last_index, 'enter_tag'] = f"LLM_{conf}_{decision['market_sentiment']}_{reason_brief}"
            
            # Update trade history
            self.update_trade_history(
                pair=metadata['pair'],
                decision=decision,
                is_entry=True,
                price=dataframe['close'].iloc[-1]
            )

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Generate exit signals based on LLM analysis
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_tag'] = ''
        
        # Check if we should analyze this pair at this time
        if not self.should_analyze_market(metadata['pair'], dataframe):
            return dataframe
            
        # Get LLM trading decision
        decision = self.current_analysis.get(metadata['pair'])
        
        # If we don't have a stored decision, create one
        if not decision:
            decision = self.create_trading_decision(dataframe, metadata)
            self.current_analysis[metadata['pair']] = decision
        
        # Generate exit signal if the LLM recommends a sell with sufficient confidence
        last_index = len(dataframe) - 1
        
        if decision['action'] == 'sell' and decision['confidence'] >= 0.7:
            dataframe.loc[last_index, 'exit_long'] = 1
            
            # Set custom tag with decision details
            conf = int(decision['confidence'] * 100)
            reason_brief = decision.get('reasoning', '')[:20].replace(' ', '_')
            dataframe.loc[last_index, 'exit_tag'] = f"LLM_{conf}_{decision['market_sentiment']}_{reason_brief}"
            
            # Update trade history
            self.update_trade_history(
                pair=metadata['pair'],
                decision=decision,
                is_entry=False,
                price=dataframe['close'].iloc[-1]
            )

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss using LLM recommendation
        """
        # Get the stored trade info
        trade_info = self.trading_memory.get(pair, {}).get('last_trade', {})
        
        if trade_info and 'stop_loss' in trade_info and trade_info['stop_loss'] is not None:
            # Return the stop loss recommended by the LLM
            return float(trade_info['stop_loss'])
            
        # Fall back to the default
        return self.stoploss

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                   current_profit: float, **kwargs) -> bool:
        """
        Custom exit logic based on take profit recommended by LLM
        """
        # Get the stored trade info
        trade_info = self.trading_memory.get(pair, {}).get('last_trade', {})
        
        if trade_info and 'take_profit' in trade_info and trade_info['take_profit'] is not None:
            # Check if current profit exceeds the recommended take profit
            take_profit = float(trade_info['take_profit'])
            
            if current_profit >= take_profit:
                logger.info(f"Taking profit for {pair} at {current_profit}% (target: {take_profit}%)")
                return True
                
        return False 

    def process_scheduled_checks(self, dataframe: DataFrame, metadata: dict) -> bool:
        """
        Process scheduled entry checks to determine if we should perform analysis now
        """
        pair = metadata['pair']
        current_price = dataframe['close'].iloc[-1]
        current_time = datetime.now()
        
        # If no scheduled entry for this pair, return False
        if pair not in self.scheduled_entries:
            return False
            
        scheduled_check = self.scheduled_entries[pair]
        should_analyze = False
        
        # Check if scheduled by price
        if scheduled_check['type'] == 'PRICE':
            target_price = float(scheduled_check['value'])
            # Check if price crossed the threshold (either direction)
            last_price = dataframe['close'].iloc[-2] if len(dataframe) > 1 else None
            
            if last_price is not None:
                # If we were waiting for price to go up
                if target_price > last_price and current_price >= target_price:
                    should_analyze = True
                    logger.info(f"Scheduled price target reached for {pair}: {target_price}")
                # If we were waiting for price to go down
                elif target_price < last_price and current_price <= target_price:
                    should_analyze = True
                    logger.info(f"Scheduled price target reached for {pair}: {target_price}")
        
        # Check if scheduled by datetime
        elif scheduled_check['type'] == 'DATETIME':
            target_time = scheduled_check['value']
            if isinstance(target_time, str):
                try:
                    target_time = datetime.fromisoformat(target_time)
                except:
                    target_time = None
                    
            if target_time and current_time >= target_time:
                should_analyze = True
                logger.info(f"Scheduled time reached for {pair}: {target_time}")
        
        # Check if scheduled by candle count
        elif scheduled_check['type'] == 'CANDLES':
            candles_waited = scheduled_check.get('candles_waited', 0) + 1
            target_candles = int(scheduled_check['value'])
            
            # Update the counter
            self.scheduled_entries[pair]['candles_waited'] = candles_waited
            
            if candles_waited >= target_candles:
                should_analyze = True
                logger.info(f"Scheduled candle count reached for {pair}: {target_candles}")
        
        # Check if scheduled by indicator value
        elif scheduled_check['type'] == 'INDICATOR':
            indicator = scheduled_check.get('indicator')
            target_value = float(scheduled_check.get('value', 0))
            operator = scheduled_check.get('operator', '>')
            
            if indicator and indicator in dataframe.columns:
                current_value = dataframe[indicator].iloc[-1]
                
                if operator == '>' and current_value > target_value:
                    should_analyze = True
                    logger.info(f"Indicator {indicator} crossed above {target_value} for {pair}")
                elif operator == '<' and current_value < target_value:
                    should_analyze = True
                    logger.info(f"Indicator {indicator} crossed below {target_value} for {pair}")
                elif operator == '=' and abs(current_value - target_value) < 0.001:
                    should_analyze = True
                    logger.info(f"Indicator {indicator} reached {target_value} for {pair}")
        
        # If we should analyze, remove the scheduled entry
        if should_analyze:
            del self.scheduled_entries[pair]
            
        return should_analyze 

    def update_risk_parameters(self, trade_result: Dict = None):
        """
        Update risk management parameters based on trading results
        and market conditions
        """
        # When a new trade result is provided
        if trade_result:
            self.risk_manager['confidence'] += 0.01  # Increase confidence with each trade
            
            # Update win rate
            total_trades = self.global_state['model_performance']['total_trades']
            if total_trades > 0:
                win_rate = self.global_state['model_performance']['successful_trades'] / total_trades
                # Smooth update of win rate (exponential moving average style)
                self.risk_manager['win_rate'] = (
                    0.9 * self.risk_manager['win_rate'] + 
                    0.1 * win_rate
                )
            
            # Update average win and loss percentage
            if trade_result.get('profit_pct', 0) > 0:
                # It's a winning trade
                if self.risk_manager['avg_win_pct'] == 0:
                    self.risk_manager['avg_win_pct'] = trade_result['profit_pct']
                else:
                    self.risk_manager['avg_win_pct'] = (
                        0.9 * self.risk_manager['avg_win_pct'] + 
                        0.1 * trade_result['profit_pct']
                    )
            else:
                # It's a losing trade
                loss_pct = abs(trade_result['profit_pct'])
                if self.risk_manager['avg_loss_pct'] == 0:
                    self.risk_manager['avg_loss_pct'] = loss_pct
                else:
                    self.risk_manager['avg_loss_pct'] = (
                        0.9 * self.risk_manager['avg_loss_pct'] + 
                        0.1 * loss_pct
                    )
                
            # Calculate expected value
            if self.risk_manager['avg_loss_pct'] > 0:
                self.risk_manager['expected_value'] = (
                    self.risk_manager['win_rate'] * self.risk_manager['avg_win_pct'] - 
                    (1 - self.risk_manager['win_rate']) * self.risk_manager['avg_loss_pct']
                )
        
        # Update market risk factor periodically (every day)
        current_time = datetime.now()
        if (current_time - self.risk_manager['last_update']).total_seconds() > 86400:  # 24 hours
            # Calculate market risk factor based on recent volatility and trend strength
            # This would ideally analyze market-wide data, for now we'll use a simple approach
            
            # Get a sample pair to analyze market conditions
            if self.dp:
                try:
                    pairs = self.dp.current_whitelist()
                    if pairs:
                        # Use BTC/USD or the first pair as a proxy for market conditions
                        market_pair = 'BTC/USDT'
                        if market_pair not in pairs:
                            market_pair = pairs[0]
                            
                        # Get daily data for market analysis
                        market_data = self.dp.get_pair_dataframe(pair=market_pair, timeframe='1d')
                        
                        if len(market_data) > 20:
                            # Calculate volatility as normalized ATR
                            market_data['atr'] = ta.ATR(market_data, timeperiod=14)
                            current_volatility = market_data['atr'].iloc[-1] / market_data['close'].iloc[-1]
                            avg_volatility = market_data['atr'].iloc[-20:].mean() / market_data['close'].iloc[-20:].mean()
                            
                            # Volatility risk factor (higher volatility = higher risk)
                            volatility_factor = current_volatility / avg_volatility if avg_volatility > 0 else 1.0
                            
                            # Trend strength factor
                            market_data['ema_20'] = ta.EMA(market_data, timeperiod=20)
                            market_data['ema_100'] = ta.EMA(market_data, timeperiod=100)
                            
                            # If price is above EMAs, reduce risk; if below, increase risk
                            current_price = market_data['close'].iloc[-1]
                            ema_20 = market_data['ema_20'].iloc[-1]
                            ema_100 = market_data['ema_100'].iloc[-1]
                            
                            trend_factor = 1.0
                            if current_price > ema_20 and current_price > ema_100:
                                trend_factor = 0.8  # Bullish trend, reduce risk
                            elif current_price < ema_20 and current_price < ema_100:
                                trend_factor = 1.2  # Bearish trend, increase risk
                                
                            # Combine factors
                            self.risk_manager['market_risk_factor'] = volatility_factor * trend_factor
                            
                            # Cap the risk factor
                            self.risk_manager['market_risk_factor'] = max(0.5, min(2.0, self.risk_manager['market_risk_factor']))
                except Exception as e:
                    logger.error(f"Error updating market risk factor: {e}")
                    
            self.risk_manager['last_update'] = current_time
            
        return self.risk_manager
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               confidence: float = 0.7) -> float:
        """
        Calculate optimal position size based on risk parameters
        and model confidence
        """
        # Update risk parameters first
        risk_params = self.update_risk_parameters()
        
        # Calculate risk per trade based on expected value and confidence
        if risk_params['expected_value'] > 0:
            # If our trading has positive expected value, we can be more aggressive
            risk_pct = risk_params['max_risk_per_trade'] * (1 + risk_params['expected_value'] / 10)
        else:
            # If our trading has negative expected value, be more conservative
            risk_pct = risk_params['max_risk_per_trade'] * (1 + risk_params['expected_value'] / 5)
            
        # Adjust risk based on market conditions
        risk_pct = risk_pct / risk_params['market_risk_factor']
        
        # Adjust risk based on model confidence
        risk_pct = risk_pct * confidence
        
        # Ensure risk stays within reasonable bounds
        risk_pct = max(0.005, min(0.05, risk_pct))  # Between 0.5% and 5%
        
        # Calculate position size based on stop loss distance
        if stop_loss > 0 and entry_price > 0:
            stop_distance_pct = abs((entry_price - stop_loss) / entry_price)
            if stop_distance_pct > 0:
                position_size = risk_pct / stop_distance_pct
                # Cap position size
                return min(1.0, max(0.1, position_size))
                
        # If we can't calculate based on stop loss, use confidence
        return min(1.0, max(0.1, confidence)) 

    def detect_market_regime(self, dataframe: DataFrame = None, pair: str = 'BTC/USDT'):
        """
        Detect the current market regime using simple heuristics based on volatility and trend
        """
        if not self.dp:
            return "unknown"
            
        try:
            # Use provided dataframe or get market data
            if dataframe is None:
                try:
                    # Try to get BTC/USDT daily data as a market proxy
                    market_data = self.dp.get_pair_dataframe(pair=pair, timeframe='1d')
                except:
                    # Fallback to any available pair
                    pairs = self.dp.current_whitelist()
                    if not pairs:
                        return "unknown"
                    market_data = self.dp.get_pair_dataframe(pair=pairs[0], timeframe='1d')
            else:
                market_data = dataframe.copy()
                
            if len(market_data) < 30:
                return "unknown"
                
            # Calculate volatility (ATR relative to price)
            market_data['atr'] = ta.ATR(market_data, timeperiod=14)
            market_data['volatility'] = market_data['atr'] / market_data['close'] * 100
            
            # Calculate trend indicators
            market_data['sma_20'] = ta.SMA(market_data['close'], timeperiod=20)
            market_data['sma_100'] = ta.SMA(market_data['close'], timeperiod=100)
            
            # Get current values
            current_volatility = market_data['volatility'].iloc[-1]
            avg_volatility = market_data['volatility'].iloc[-20:].mean()
            
            current_price = market_data['close'].iloc[-1]
            sma_20 = market_data['sma_20'].iloc[-1]
            sma_100 = market_data['sma_100'].iloc[-1]
            
            # Determine regime
            if current_price > sma_20 and sma_20 > sma_100:
                trend = "bullish"
            elif current_price < sma_20 and sma_20 < sma_100:
                trend = "bearish"
            else:
                trend = "sideways"
                
            if current_volatility > avg_volatility * 1.3:
                volatility = "high_volatility"
            else:
                volatility = "normal_volatility"
                
            regime = f"{trend}_{volatility}"
            
            # Update global state
            current_time = datetime.now()
            previous_regime = self.global_state['market_regime']['current']
            
            if regime != previous_regime and previous_regime != 'unknown':
                # Record regime change
                self.global_state['market_regime']['history'].append({
                    'from': previous_regime,
                    'to': regime,
                    'time': current_time
                })
                
                # Keep only last 10 regime changes
                if len(self.global_state['market_regime']['history']) > 10:
                    self.global_state['market_regime']['history'] = self.global_state['market_regime']['history'][-10:]
                    
                logger.info(f"Market regime changed from {previous_regime} to {regime}")
                    
            self.global_state['market_regime']['current'] = regime
            self.global_state['market_regime']['last_update'] = current_time
            
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "unknown" 