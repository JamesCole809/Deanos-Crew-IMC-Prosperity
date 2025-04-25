import math
import numpy as np  # type: ignore
import string
import jsonpickle
import json
from math import log, sqrt
from statistics import NormalDist
from typing import List, Dict, Tuple, Any
from datamodel import (
    OrderDepth,
    TradingState,
    Order,
    Listing,
    Observation,
    ProsperityEncoder,
    Symbol,
    Trade,
)


# Configure which trader to copy for each asset with optional counterparty, quantity filters, and signal weights
COPY_TARGETS = {
    "SQUID_INK": {
        "signals": [
            {"trader": "Olivia", "counterparty": "", "quantity": 0, "weight": 2, "watch_asset": "SQUID_INK"},
            # Medium signal
        ]
    },
    "CROISSANTS": {
        "signals": [
            {"trader": "Olivia", "counterparty": "", "quantity": 0, "weight": 1.0, "watch_asset": "CROISSANTS"}
        ]
    },
    "VOLCANIC_ROCK": {
        "signals": [
            {"trader": "Pablo", "counterparty": "", "quantity": 0, "weight": 2, "watch_asset": "VOLCANIC_ROCK"}
        ]
    },
    "VOLCANIC_ROCK_VOUCHER_9500": {
        "signals": [
            {"trader": "Pablo", "counterparty": "", "quantity": 0, "weight": 2, "watch_asset": "VOLCANIC_ROCK"}
        ]
    },
    
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
                     (log(spot) - log(strike)) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
                volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
                     (log(spot) - log(strike)) + (0.5 * volatility * volatility) * time_to_expiry
             ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(
            call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-15
    ):
        """
        A binary-search approach to implied vol.
        We'll exit once we get close to the observed call_price,
        or we run out of iterations.
        """
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Status:
    def __init__(self, product: str, state: TradingState, strike=None) -> None:
        self.product = product
        self._state = state
        self.ma_window = 10  # For the moving average of implied vol
        self.alpha = 0.3
        self.volatility = 0.16
        self.initial_time_to_expiry = 6  # 6 trading days remaining
        self.strike = strike
        self.price_history = []  # Track price history for trend analysis
        self.volatility_history = []  # Track volatility history
        self.last_trade_timestamp = 0  # Track when we last traded
        self.trade_count = 0  # Count trades to adjust aggressiveness
        self.profit_history = []  # Track profit/loss history

    @property
    def order_depth(self) -> OrderDepth:
        return self._state.order_depths[self.product]

    @property
    def bids(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())

    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)

    @property
    def possible_buy_amt(self) -> int:
        return 50 - self.position

    @property
    def possible_sell_amt(self) -> int:
        return 50 + self.position

    @property
    def jam_possible_buy_amt(self) -> int:
        return 350 - self.position

    @property
    def jam_possible_sell_amt(self) -> int:
        return 350 + self.position

    @property
    def best_bid(self) -> int:
        bids = self._state.order_depths[self.product].buy_orders
        return max(bids.keys()) if bids else 0

    @property
    def best_ask(self) -> int:
        asks = self._state.order_depths[self.product].sell_orders
        return min(asks.keys()) if asks else float('inf')

    @property
    def maxamt_midprc(self) -> float:
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders
        if not buy_orders or not sell_orders:
            return (self.best_bid + self.best_ask) / 2.0
        max_bv = 0
        max_bv_price = self.best_bid
        for p, v in buy_orders.items():
            if v > max_bv:
                max_bv = v
                max_bv_price = p
        max_sv = 0
        max_sv_price = self.best_ask
        for p, v in sell_orders.items():
            if -v > max_sv:
                max_sv = -v
                max_sv_price = p
        return (max_bv_price + max_sv_price) / 2

    @property
    def vwap(self) -> float:
        """
        Calculate Volume Weighted Average Price (VWAP) for the product.
        Combines bid and ask data.
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        total_value = 0  # Total (price * volume)
        total_volume = 0  # Total volume

        # Aggregate bid data
        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        # Aggregate ask data
        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        # Prevent division by zero
        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0  # Default to mid-price

        return total_value / total_volume

    @property
    def timestamp(self) -> int:
        return self._state.timestamp

    @property
    def order_depth(self) -> OrderDepth:
        return self._state.order_depths[self.product]

    @property
    def bids(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].buy_orders.items())

    @property
    def asks(self) -> List[Tuple[int, int]]:
        return list(self._state.order_depths[self.product].sell_orders.items())

    @property
    def position(self) -> int:
        return self._state.position.get(self.product, 0)

    @property
    def possible_buy_amt(self) -> int:
        """
        The position limit is different for each product.
        We keep the logic that KELP is +/-50, vouchers are +/-200,
        and VOLCANIC_ROCK is +/-400.
        """
        if self.product == "KELP":
            return 50 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9500":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9750":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10000":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10250":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10500":
            return 200 - self.position
        elif self.product == "VOLCANIC_ROCK":
            return 400 - self.position

    @property
    def possible_sell_amt(self) -> int:
        if self.product == "KELP":
            return 50 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9500":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_9750":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10000":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10250":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK_VOUCHER_10500":
            return 200 + self.position
        elif self.product == "VOLCANIC_ROCK":
            return 400 + self.position

    @property
    def best_bid(self) -> int:
        bids = self._state.order_depths[self.product].buy_orders
        return max(bids.keys()) if bids else 0

    @property
    def best_ask(self) -> int:
        asks = self._state.order_depths[self.product].sell_orders
        return min(asks.keys()) if asks else float("inf")

    @property
    def vwap(self) -> float:
        """
        Compute the VWAP, combining all current bid/ask levels.
        This is a safer reference point than the naive mid-price.
        """
        buy_orders = self._state.order_depths[self.product].buy_orders
        sell_orders = self._state.order_depths[self.product].sell_orders

        total_value = 0
        total_volume = 0

        for price, volume in buy_orders.items():
            total_value += price * volume
            total_volume += volume

        for price, volume in sell_orders.items():
            total_value += price * abs(volume)
            total_volume += abs(volume)

        if total_volume == 0:
            return (self.best_bid + self.best_ask) / 2.0
        return total_value / total_volume

    def update_price_history(self, price: float) -> None:
        """Update price history with current price"""
        self.price_history.append(price)
        # Keep only the last 20 prices
        if len(self.price_history) > 20:
            self.price_history = self.price_history[-20:]

    def update_volatility_history(self, volatility: float) -> None:
        """Update volatility history"""
        self.volatility_history.append(volatility)
        # Keep only the last 10 volatility readings
        if len(self.volatility_history) > 10:
            self.volatility_history = self.volatility_history[-10:]

    def update_profit_history(self, profit: float) -> None:
        """Update profit history"""
        self.profit_history.append(profit)
        # Keep only the last 5 profit readings
        if len(self.profit_history) > 5:
            self.profit_history = self.profit_history[-5:]

    def get_price_trend(self) -> float:
        """Calculate price trend based on recent history"""
        if len(self.price_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.price_history)))
        y = self.price_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def get_volatility_trend(self) -> float:
        """Calculate volatility trend"""
        if len(self.volatility_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.volatility_history)))
        y = self.volatility_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def get_recent_profit_trend(self) -> float:
        """Calculate recent profit trend"""
        if len(self.profit_history) < 2:
            return 0

        # Simple linear regression for trend
        x = list(range(len(self.profit_history)))
        y = self.profit_history

        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)

        # Calculate slope
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0

        slope = numerator / denominator
        return slope

    def update_IV_history(self, underlying_price) -> None:
        """Refresh the stored implied vol reading."""
        temp_history = IV_history.get_IV_history(self.product)
        temp_history.append(self.IV(underlying_price))
        IV_history.set_IV_history(self.product, temp_history[-self.ma_window:])

        # Also update our volatility history
        self.update_volatility_history(self.IV(underlying_price))

    def IV(self, underlying_price) -> float:
        return BlackScholes.implied_volatility(
            call_price=self.vwap,
            spot=underlying_price,
            strike=self.strike,
            time_to_expiry=self.tte,
        )

    def moving_average(self, underlying_price: int) -> float:
        """
        Simple average of the last few implied vol readings.
        If we have no stored history yet, just seed from current IV.
        """
        hist = IV_history.get_IV_history(self.product)
        if not hist:
            return self.IV(underlying_price)
        return sum(hist) / len(hist)

    @property
    def tte(self) -> float:
        """
        We have 6 days left to expiry. Each "day" is effectively chunked
        as you proceed in your simulation. We divide by 250 so that
        each day is treated as 1 "trading day" in annualized terms.
        """
        # The environment's timestamp goes up to ~1,000,000 per day,
        # so we do (6 - dayProgress) / 250.
        return (self.initial_time_to_expiry - (self.timestamp / 1_000_000)) / 252.0


class IV_history:
    def __init__(self):
        self.v_9500_IV_history = []
        self.v_9750_IV_history = []
        self.v_10000_IV_history = []
        self.v_10250_IV_history = []
        self.v_10500_IV_history = []

    def get_IV_history(self, product: str) -> List[float]:
        if product == "VOLCANIC_ROCK_VOUCHER_9500":
            return self.v_9500_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            return self.v_9750_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            return self.v_10000_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            return self.v_10250_IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            return self.v_10500_IV_history
        return []

    def set_IV_history(self, product: str, IV_history: List[float]) -> None:
        if product == "VOLCANIC_ROCK_VOUCHER_9500":
            self.v_9500_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_9750":
            self.v_9750_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10000":
            self.v_10000_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10250":
            self.v_10250_IV_history = IV_history
        elif product == "VOLCANIC_ROCK_VOUCHER_10500":
            self.v_10500_IV_history = IV_history


IV_history = IV_history()


class Product:
    RAINFOREST_RESIN = 'RAINFOREST_RESIN'
    KELP = 'KELP'
    SQUID_INK = 'SQUID_INK'
    CROISSANTS = 'CROISSANTS'
    JAMS = 'JAMS'
    DJEMBES = 'DJEMBES'
    PICNIC_BASKET1 = 'PICNIC_BASKET1'
    PICNIC_BASKET2 = 'PICNIC_BASKET2'
    SYNTHETIC = "SYNTHETIC"
    # added from round3Kai:
    VOLCANIC_ROCK = 'VOLCANIC_ROCK'
    VOLCANIC_ROCK_VOUCHER_9500 = 'VOLCANIC_ROCK_VOUCHER_9500'
    VOLCANIC_ROCK_VOUCHER_9750 = 'VOLCANIC_ROCK_VOUCHER_9750'
    VOLCANIC_ROCK_VOUCHER_10000 = 'VOLCANIC_ROCK_VOUCHER_10000'
    VOLCANIC_ROCK_VOUCHER_10250 = 'VOLCANIC_ROCK_VOUCHER_10250'
    VOLCANIC_ROCK_VOUCHER_10500 = 'VOLCANIC_ROCK_VOUCHER_10500'


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 0,
        "join_edge": 4,
        "default_edge": 2,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 10,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 2,
        "manage_position": True,
        "soft_position_limit": 10,
    },
    Product.PICNIC_BASKET1: {
        "default_spread_mean": 48.762433,
        "default_spread_std": 85.119451,
        "spread_std_window": 100,
        "z_score_threshold": 10,
        "z_score_close_threshold": 0.1,
        "target_position": 60,
    },
    Product.PICNIC_BASKET2: {
        "default_spread_mean": 30.235967,
        "default_spread_std": 59.849200,
        "spread_std_window": 50,
        "z_score_threshold": 15,
        "target_position": 100,
        "z_score_close_threshold": 0.2,
    }
}

BASKET1_WEIGHTS = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBES: 1,
}
BASKET2_WEIGHTS = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}


# -------------------------------------------------------------------
# "Trade" class with static methods for each product or approach.
# -------------------------------------------------------------------
class Trade:
    """Trading strategies for different products."""

    @staticmethod
    def kelp(state: Status) -> List[Order]:
        """
        KELP is just a demonstration: we do a quote-based approach
        to buy below fair_value, sell above fair_value.
        Not strictly relevant to the VOUCHER logic.
        """
        orders = []
        AGGRESSOR = False
        QUOTER = True
        vwap_price = state.vwap
        fair_value = vwap_price

        if AGGRESSOR:
            # Omitted for brevity...
            pass

        if QUOTER:
            if state.bids:
                best_bid_level = max(state.bids, key=lambda x: x[1])
                bid_price, bid_vol = best_bid_level
                buy_qty = min(state.possible_buy_amt, bid_vol)
                if buy_qty > 0 and bid_price < fair_value:
                    orders.append(Order(state.product, int(bid_price + 1), buy_qty))

            if state.asks:
                best_ask_level = max(state.asks, key=lambda x: abs(x[1]))
                ask_price, ask_vol = best_ask_level
                sell_qty = min(state.possible_sell_amt, abs(ask_vol))
                if sell_qty > 0 and ask_price > fair_value:
                    orders.append(Order(state.product, int(ask_price - 1), -sell_qty))

        return orders

    @staticmethod
    def hedge_deltas(
            underlying_state: Status,
            v_9500_state: Status,
            v_9750_state: Status,
            v_10000_state: Status,
            v_10250_state: Status,
            v_10500_state: Status,
    ) -> List[Order]:
        """
        Simple net-delta hedging that restricts total net delta to ±50 at all times.
        """
        orders = []

        # 1. Calculate net delta: add up (position * delta) for each voucher
        net_delta = underlying_state.position  # Start with underlying position
        for voucher_state, strike in [
            (v_9500_state, 9500),
            (v_9750_state, 9750),
            (v_10000_state, 10000),
            (v_10250_state, 10250),
            (v_10500_state, 10500),
        ]:
            if voucher_state is None:
                continue
            # Approximate delta
            delta_per_unit = BlackScholes.delta(
                spot=underlying_state.vwap,
                strike=strike,
                time_to_expiry=voucher_state.tte,
                volatility=voucher_state.volatility,
            )
            net_delta += delta_per_unit * voucher_state.position

        # 2. Check if net delta goes beyond ±50
        MAX_DELTA = 50
        if net_delta > MAX_DELTA:
            # Hedge quantity: bring net_delta *down* to +50
            hedge_quantity = int(net_delta - MAX_DELTA)  # how many to sell

            # Make sure we do not exceed what's possible to sell
            # (the underlying is "VOLCANIC_ROCK")
            # If hedge_quantity is 20, we need to SELL 20
            can_sell = underlying_state.possible_sell_amt
            hedge_quantity = min(hedge_quantity, can_sell)

            # Place an order at best_bid to SELL
            if hedge_quantity > 0 and underlying_state.best_bid > 0:
                orders.append(
                    Order(
                        symbol=underlying_state.product,
                        price=underlying_state.best_bid,
                        quantity=-hedge_quantity,  # negative => SELL
                    )
                )

        elif net_delta < -MAX_DELTA:
            # Hedge quantity: bring net_delta *up* to -50
            hedge_quantity = int(abs(net_delta + MAX_DELTA))  # how many to buy

            # Make sure we do not exceed what's possible to buy
            can_buy = underlying_state.possible_buy_amt
            hedge_quantity = min(hedge_quantity, can_buy)

            # Place an order at best_ask to BUY
            if hedge_quantity > 0 and underlying_state.best_ask < float("inf"):
                orders.append(
                    Order(
                        symbol=underlying_state.product,
                        price=underlying_state.best_ask,
                        quantity=hedge_quantity,  # positive => BUY
                    )
                )

        # 3. If net_delta is between -50 and +50, do nothing (already within range)
        return orders



    @staticmethod
    def volcanic_rock(state: TradingState) -> List[Order]:
        """Generate trading orders for VOLCANIC_ROCK using a conservative mean-reversion strategy."""
        orders: List[Order] = []

        # Get current market data
        product = "VOLCANIC_ROCK"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        # Calculate basic metrics
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = (
            min(order_depth.sell_orders.keys())
            if order_depth.sell_orders
            else float("inf")
        )
        mid_price = (
            (best_bid + best_ask) / 2 if best_bid and best_ask != float("inf") else None
        )

        if not mid_price:
            return orders

        # Calculate VWAP for mean reversion
        vwap = (
            sum(price * abs(qty) for price, qty in order_depth.buy_orders.items())
            / sum(abs(qty) for qty in order_depth.buy_orders.values())
            if order_depth.buy_orders
            else mid_price
        )

        # Conservative position limits
        max_position = 20  # Reduced from previous value
        min_position = -20

        # Calculate price deviation from VWAP
        price_deviation = (mid_price - vwap) / vwap

        # Trading thresholds
        entry_threshold = 0.002  # 0.2% deviation for entry
        exit_threshold = 0.001  # 0.1% deviation for exit

        # Position sizing based on deviation
        base_quantity = 5  # Reduced base quantity
        quantity = min(base_quantity, max_position - position)

        # Trading logic
        if price_deviation > entry_threshold and position < max_position:
            # Price is above VWAP - consider selling
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -quantity))

        elif price_deviation < -entry_threshold and position > min_position:
            # Price is below VWAP - consider buying
            if best_bid > 0:
                orders.append(Order(product, best_bid, quantity))

        # Exit logic - more aggressive
        if position > 0 and price_deviation > exit_threshold:
            # Exit long position if price is above VWAP
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -position))

        elif position < 0 and price_deviation < -exit_threshold:
            # Exit short position if price is below VWAP
            if best_bid > 0:
                orders.append(Order(product, best_bid, -position))

        return orders

    @staticmethod
    def volcanic_rock_copy(state: TradingState) -> List[Order]:
        """Trading strategy for VOLCANIC_ROCK"""
        orders = []
        
        # Get bias from Pablo's trades
        bias = 0.0
        if 'VOLCANIC_ROCK' in state.market_trades:
            for trade in state.market_trades['VOLCANIC_ROCK']:
                if trade.buyer == 'Pablo':
                    bias = min(1.0, bias + 0.2)  # Increase bullish bias
                elif trade.seller == 'Pablo':
                    bias = max(-1.0, bias - 0.2)  # Increase bearish bias

        product = "VOLCANIC_ROCK"
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)

        # Calculate basic metrics
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask != float("inf") else None

        if not mid_price:
            return orders

        # Calculate VWAP
        vwap = (
            sum(price * abs(qty) for price, qty in order_depth.buy_orders.items())
            / sum(abs(qty) for qty in order_depth.buy_orders.values())
            if order_depth.buy_orders
            else mid_price
        )

        # Adjust position limits based on bias
        max_position = int(400 * (1 + abs(bias) * 0.2))  # Increase limit when confident
        min_position = -max_position

        # Calculate price deviation from VWAP
        price_deviation = (mid_price - vwap) / vwap

        # Adjust thresholds based on bias
        base_entry = 0.002
        base_exit = 0.001
        
        if bias > 0:  # Bullish
            entry_threshold = base_entry * (1 - bias * 0.5)  # Lower threshold for buys
            exit_threshold = base_exit * (1 + bias * 0.5)   # Higher threshold for sells
        else:  # Bearish
            entry_threshold = base_entry * (1 + abs(bias) * 0.5)  # Higher threshold for buys
            exit_threshold = base_exit * (1 - abs(bias) * 0.5)   # Lower threshold for sells

        # Trading logic with bias adjustments
        base_quantity = int(5 * (1 + abs(bias)))  # Increase size with confidence
        quantity = min(base_quantity, max_position - position)

        if price_deviation > entry_threshold and position < max_position:
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -quantity))

        elif price_deviation < -entry_threshold and position > min_position:
            if best_bid > 0:
                orders.append(Order(product, best_bid, quantity))

        # Exit logic
        if position > 0 and price_deviation > exit_threshold:
            if best_ask < float("inf"):
                orders.append(Order(product, best_ask, -position))

        elif position < 0 and price_deviation < -exit_threshold:
            if best_bid > 0:
                orders.append(Order(product, best_bid, -position))
        print("volcanic_rock_copy orders", orders)
        return orders
    
    @staticmethod
    def voucher(state: Status, underlying_state: Status, strike: int) -> List[Order]:
        """
        Simplified volatility trading for 9750 and 10000 strikes
        """
        orders = []

        # Only trade 9750 and 10000 strikes
        if strike not in [9750, 10000]:
            return orders

        # Update price and volatility history
        state.update_IV_history(underlying_state.vwap)
        state.update_price_history(state.vwap)

        # Calculate current and previous implied volatility
        current_IV = BlackScholes.implied_volatility(
            call_price=state.vwap,
            spot=underlying_state.vwap,
            strike=strike,
            time_to_expiry=state.tte,
        )
        prev_IV = state.moving_average(underlying_state.vwap)

        # Get market trends
        price_trend = state.get_price_trend()
        vol_trend = state.get_volatility_trend()
        profit_trend = state.get_recent_profit_trend()

        # Base parameters
        base_threshold = 0.002
        max_position = 200

        # Adjust threshold based on volatility trend
        threshold = base_threshold
        if vol_trend > 0.01:
            threshold *= 1.2
        elif vol_trend < -0.01:
            threshold *= 0.8

        # Selling volatility (when current IV > previous IV + threshold)
        if current_IV > prev_IV + threshold:
            if state.bids and state.position > -max_position:
                # Calculate position room and base quantity
                position_room = max_position + state.position
                base_quantity = min(state.possible_sell_amt, state.bids[0][1])
                quantity = min(base_quantity, position_room)

                # Adjust quantity based on market conditions
                if abs(price_trend) > 0.1:
                    quantity = int(quantity * 0.7)
                if vol_trend > 0.01:
                    quantity = int(quantity * 0.8)
                if profit_trend < -1000:
                    quantity = int(quantity * 0.6)

                # Place sell order if quantity is positive
                if quantity > 0:
                    orders.append(Order(state.product, state.best_bid, -quantity))
                    state.last_trade_timestamp = state.timestamp
                    state.trade_count += 1

        # Buying volatility (when current IV < previous IV - threshold)
        elif current_IV < prev_IV - threshold:
            if state.asks and state.position < max_position:
                # Calculate position room and base quantity
                position_room = max_position - state.position
                base_quantity = min(state.possible_buy_amt, abs(state.asks[0][1]))
                quantity = min(base_quantity, position_room)

                # Adjust quantity based on market conditions
                if abs(price_trend) > 0.1:
                    quantity = int(quantity * 0.7)
                if vol_trend > 0.01:
                    quantity = int(quantity * 0.8)
                if profit_trend < -1000:
                    quantity = int(quantity * 0.6)

                # Place buy order if quantity is positive
                if quantity > 0:
                    orders.append(Order(state.product, state.best_ask, quantity))
                    state.last_trade_timestamp = state.timestamp
                    state.trade_count += 1

        return orders


class Trader:
    def __init__(self, params=None):
        self.position = 0
        self.desired_position_pct = {asset: 0.0 for asset in COPY_TARGETS.keys()}
        self.position_limit = 75
        self.sunlight_window = []  # Keep track of recent sunlight values
        self.window_size = 10  # Increased to 10 for slope calculation
        self.sunlight_threshold = 49  # Only trade when above this level
        self.slope_threshold = 0.004  # Minimum slope to trigger position close
        self.last_trend = None
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {Product.RAINFOREST_RESIN: 50,
                      Product.KELP: 50,
                      Product.SQUID_INK: 50,
                      Product.CROISSANTS: 250,
                      Product.JAMS: 350,
                      Product.DJEMBES: 60,
                      Product.PICNIC_BASKET1: 60,
                      Product.PICNIC_BASKET2: 100,
                      Product.VOLCANIC_ROCK: 200,
                      Product.VOLCANIC_ROCK_VOUCHER_9500: 200,
                      }

    def should_copy_trade(self, trade, target_config, actual_trader):
        """Determine if we should copy this trade based on filters"""
        # Check if it's our target trader
        if trade.buyer != actual_trader and trade.seller != actual_trader:
            return False

        # Check counterparty filter if specified
        if target_config["counterparty"]:
            if trade.buyer == actual_trader and trade.seller != target_config["counterparty"]:
                return False
            if trade.seller == actual_trader and trade.buyer != target_config["counterparty"]:
                return False

        # Check exact quantity match if specified (quantity = 0 means copy any quantity)
        if target_config["quantity"] > 0 and trade.quantity != target_config["quantity"]:
            return False

        return True

    def adjust_desired_positions(self,state: TradingState, asset: str):
        for signal_config in COPY_TARGETS[asset]["signals"]:
            target_trader = signal_config["trader"]
            inverse_copy = target_trader.startswith("-")
            actual_trader = target_trader[1:] if inverse_copy else target_trader
            watch_asset = signal_config["watch_asset"]

            trades = []
            try:
                trades = state.market_trades[watch_asset]
            except:
                continue

            for trade in trades:
                if self.should_copy_trade(trade, signal_config, actual_trader):
                    if trade.buyer == actual_trader:
                        # If they buy: normal copy → positive adjustment, inverse copy → negative adjustment
                        adjustment = -signal_config["weight"] if inverse_copy else signal_config["weight"]
                        self.desired_position_pct[asset] = min(1.0, max(-1.0,
                                                                        self.desired_position_pct[asset] + adjustment))
                    if trade.seller == actual_trader:
                        # If they sell: normal copy → negative adjustment, inverse copy → positive adjustment
                        adjustment = signal_config["weight"] if inverse_copy else -signal_config["weight"]
                        self.desired_position_pct[asset] = min(1.0, max(-1.0,self.desired_position_pct[asset] + adjustment))


    def calculate_slope(self, values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        y = np.array(values)
        slope, _ = np.polyfit(x, y, 1)
        return slope

    def make_orders(
            self,
            product,
            order_depth: OrderDepth,
            fair_value: float,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            disregard_edge: float,  # disregard trades within this edge for pennying or joining
            join_edge: float,  # join trades within this edge
            default_edge: float,  # default edge to request if there are no levels to penny or join
            manage_position: bool = False,
            soft_position_limit: int = 0,
            # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # join
            else:
                bid = best_bid_below_fair + 1  # penny

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def market_make(
            self,
            product: str,
            orders: List[Order],
            bid: int,
            ask: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order

        return buy_order_volume, sell_order_volume

    def take_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def take_best_orders(
            self,
            product: str,
            fair_value: int,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
    ):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)  # Max amount to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)  # Max amount to sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def clear_orders(
            self,
            product: str,
            order_depth: OrderDepth,
            fair_value: float,
            clear_width: int,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_position_order(
            self,
            product: str,
            fair_value: float,
            width: int,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                        last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_best_squid_orders(
            self,
            product: str,
            vwap: float,
            take_width: float,
            orders: List[Order],
            order_depth: OrderDepth,
            position: int,
            buy_order_volume: int,
            sell_order_volume: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
            # New inputs for cross-product signals:
            kelp_order_depth: OrderDepth = None,
            baseline_mm_volume: int = 26,  # Typical market maker volume
            mm_tolerance: float = 0.3  # Tolerance factor (30%)
    ):

        position_limit = self.LIMIT[product]

        # ---------------------------
        # Detect cross-market signals from kelp.
        anomaly_bullish = False
        anomaly_bearish = False

        # Check for bullish signal using kelp ask side.
        if kelp_order_depth is not None and len(kelp_order_depth.sell_orders) != 0:
            kelp_best_ask = min(kelp_order_depth.sell_orders.keys())
            kelp_best_ask_volume = abs(kelp_order_depth.sell_orders[kelp_best_ask])
            # If the kelp ask volume is notably lower than the baseline, flag bullish anomaly.
            if kelp_best_ask_volume < baseline_mm_volume * (1 - mm_tolerance):
                anomaly_bullish = True

        # Check for bearish signal using kelp bid side.
        if kelp_order_depth is not None and len(kelp_order_depth.buy_orders) != 0:
            kelp_best_bid = max(kelp_order_depth.buy_orders.keys())
            kelp_best_bid_volume = abs(kelp_order_depth.buy_orders[kelp_best_bid])
            # If the kelp bid volume is notably lower than the baseline, flag bearish anomaly.
            if kelp_best_bid_volume < baseline_mm_volume * (1 - mm_tolerance):
                anomaly_bearish = True

        # Adjust effective thresholds based on detected signals.
        # For bullish signals, relax buy threshold; for bearish signals, relax sell threshold.
        effective_take_width_buy = take_width * (0.5 if anomaly_bullish else 1.0)
        effective_take_width_sell = take_width * (0.5 if anomaly_bearish else 1.0)

        # ---------------------------
        # Process Sell Orders (Buy in squid ink from the sell orders)
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                # When buying, check if price is below VWAP adjusted by effective_take_width_buy.
                if best_ask <= vwap - effective_take_width_buy:
                    quantity = min(best_ask_amount, position_limit - position)  # Maximum amount to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        # Process Buy Orders (Sell in squid ink into the bid orders)
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # When selling, check if price is above VWAP adjusted by effective_take_width_sell.
                if best_bid >= vwap + effective_take_width_sell:
                    quantity = min(best_bid_amount, position_limit + position)  # Maximum amount to sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                   >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                   >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("squid_ink_last_price", None) == None:
                    mm_mid_price = (best_ask + best_bid) / 2
                else:
                    mm_mid_price = traderObject["squid_ink_last_price"]
            else:
                mm_mid_price = (mm_ask + mm_bid) / 2

            if "squid_ink_trade_history" not in traderObject:
                traderObject["squid_ink_trade_history"] = []
            vol_ask = abs(order_depth.sell_orders.get(mm_ask, 0)) if mm_ask else 0
            vol_bid = abs(order_depth.buy_orders.get(mm_bid, 0)) if mm_bid else 0
            volume = min(vol_ask, vol_bid)
            traderObject["squid_ink_trade_history"].append((mm_mid_price, volume))

            max_window = 100
            if len(traderObject["squid_ink_trade_history"]) > max_window:
                traderObject["squid_ink_trade_history"].pop(0)
            trade_history = traderObject["squid_ink_trade_history"]
            vwap_numerator = sum(price * vol for price, vol in trade_history)
            vwap_denominator = sum(vol for _, vol in trade_history)
            vwap = vwap_numerator / vwap_denominator if vwap_denominator != 0 else mm_mid_price

            if traderObject.get("squid_ink_last_price", None) != None:
                last_price = traderObject["squid_ink_last_price"]
                last_returns = (mm_mid_price - last_price) / last_price
                pred_returns = (last_returns * self.params[Product.SQUID_INK]["reversion_beta"])
                fair = mm_mid_price + (mm_mid_price * pred_returns)
            else:
                fair = mm_mid_price
            traderObject["squid_ink_last_price"] = mm_mid_price
            return fair, vwap
        return None

    def take_squid_orders(
            self,
            product: str,
            vwap: int,
            order_depth: OrderDepth,
            take_width: float,
            position: int,
            prevent_adverse: bool = False,
            adverse_volume: int = 0,
            kelp_order_depth: OrderDepth = None,
            baseline_mm_volume: int = 26,
            mm_tolerance: float = 0.3
    ):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_squid_orders(
            product,
            vwap,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
            kelp_order_depth,
            baseline_mm_volume,
            mm_tolerance
        )
        return orders, buy_order_volume, sell_order_volume

    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol
        )

    def get_synthetic_basket_order_depth(
            self,
            order_depths: Dict[str, OrderDepth],
            basket_weights: Dict[str, int],
    ) -> OrderDepth:

        synthetic_order_depth = OrderDepth()
        best_bids = {}
        best_asks = {}
        for product in basket_weights:
            if product in order_depths:
                order_depth = order_depths[product]
            else:
                continue
            best_bids[product] = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
            best_asks[product] = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float("inf")

        implied_bid = sum(best_bids[product] * quantity for product, quantity in basket_weights.items())
        implied_ask = sum(best_asks[product] * quantity for product, quantity in basket_weights.items())

        if implied_bid > 0:
            bid_volumes = []
            for product, quantity in basket_weights.items():
                volume = order_depths[product].buy_orders.get(best_bids[product], 0) // quantity
                bid_volumes.append(volume)
            synthetic_order_depth.buy_orders[implied_bid] = min(bid_volumes)

        if implied_ask < float("inf"):
            ask_volumes = []
            for product, quantity in basket_weights.items():
                volume = -order_depths[product].sell_orders.get(best_asks[product], 0) // quantity
                ask_volumes.append(volume)
            synthetic_order_depth.sell_orders[implied_ask] = -min(ask_volumes)

        return synthetic_order_depth

    def convert_synthetic_basket_orders(
            self,
            synthetic_orders: List[Order],
            order_depths: Dict[str, OrderDepth],
            basket_weights: Dict[str, int]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {product: [] for product in basket_weights}

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_weights)

        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:  # Buy the basket
                component_prices = {
                    product: min(order_depths[product].sell_orders.keys(), default=float("inf"))
                    for product in basket_weights
                }

            elif quantity < 0 and price <= best_bid:  # Sell the basket
                component_prices = {
                    product: max(order_depths[product].buy_orders.keys(), default=0)
                    for product in basket_weights
                }
            else:
                continue

            for product, product_price in component_prices.items():
                component_orders[product].append(
                    Order(
                        product,
                        product_price,
                        quantity * basket_weights[product]
                    )
                )

        return component_orders

    def execute_spread_orders(
            self,
            target_position: int,
            basket_position: int,
            order_depths: Dict[str, OrderDepth],
            product: Product,
            basket_weights: Dict[str, int]
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_weights)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(synthetic_order_depth.buy_orders[synthetic_bid_price])

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(product, basket_ask_price, execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)]
        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(synthetic_order_depth.sell_orders[synthetic_ask_price])

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [Order(product, basket_bid_price, -execute_volume)]
            synthetic_orders = [Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)]

        aggregate_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths, basket_weights)
        aggregate_orders[product] = basket_orders
        return aggregate_orders

    def spread_orders(
            self,
            order_depths: Dict[str, OrderDepth],
            product: Product,
            basket_weights: Dict[str, int],
            basket_position: int,
            spread_data: Dict[str, Any],
    ):
        if product not in order_depths.keys():
            return None

        basket_order_depth = order_depths[product]
        synthetic_order_depth = self.get_synthetic_basket_order_depth(order_depths, basket_weights)

        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid

        spread_data["spread_history"].append(spread)

        spread_std_window = self.params[product]["spread_std_window"]
        if (len(spread_data["spread_history"]) < spread_std_window):
            return None
        elif (len(spread_data["spread_history"]) > spread_std_window):
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])
        default_mean = self.params[product]["default_spread_mean"]
        z_threshold = self.params[product]["z_score_threshold"]
        target_position = self.params[product]["target_position"]
        zscore = (spread - default_mean) / spread_std

        if abs(zscore) < self.params[product]["z_score_close_threshold"] and basket_position != 0:
            return self.execute_spread_orders(
                0,  # Target position is zero to close the position
                basket_position,
                order_depths,
                product,
                basket_weights,
            )

        if zscore >= z_threshold and basket_position != -target_position:
            return self.execute_spread_orders(
                -target_position,
                basket_position,
                order_depths,
                product,
                basket_weights,
            )

        if zscore <= -z_threshold and basket_position != target_position:
            return self.execute_spread_orders(
                target_position,
                basket_position,
                order_depths,
                product,
                basket_weights,
            )

        spread_data["prev_zscore"] = zscore
        return None

    def rainforest_resin_orders(
            self,
            order_depth: OrderDepth,
            fair_value: float,
            width: float,
            position: int,
            position_limit: int
    ) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # 1) pick levels to clear or penny
        higher_asks = [
            price for price in order_depth.sell_orders
            if price > fair_value + 1
        ]
        lower_bids = [
            price for price in order_depth.buy_orders
            if price < fair_value - 1
        ]
        baaf = min(higher_asks) if higher_asks else fair_value + 2
        bbbf = max(lower_bids) if lower_bids else fair_value - 2

        # 2) take any crosses
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            amt = -order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                q = min(amt, position_limit - position)
                if q > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, q))
                    buy_order_volume += q

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            amt = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                q = min(amt, position_limit + position)
                if q > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -q))
                    sell_order_volume += q

        # 3) clear stale levels via clear_position_order
        buy_order_volume, sell_order_volume = self.clear_position_order(
            "RAINFOREST_RESIN",
            fair_value,
            1,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        # 4) repost maker quotes
        buy_qty = position_limit - (position + buy_order_volume)
        sell_qty = position_limit + (position - sell_order_volume)
        if buy_qty > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_qty))
        if sell_qty > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_qty))

        return orders

    def run(self, state: TradingState):
        # 0. decode previous traderData
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)

        result: Dict[str, List[Order]] = {}

        orders: Dict[str, List[Order]] = {}
        orders["MAGNIFICENT_MACARONS"] = []
        trade_made = "NONE"

        # Get market data
        if "MAGNIFICENT_MACARONS" not in state.order_depths:
            return {}, 0, ""

        order_depth = state.order_depths["MAGNIFICENT_MACARONS"]

        # Get market prices
        normal_best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
        normal_best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')

        # Get sunlight data
        current_sunlight = state.observations.conversionObservations.get("MAGNIFICENT_MACARONS", None).sunlightIndex
        self.sunlight_window.append(current_sunlight)
        if len(self.sunlight_window) > self.window_size:
            self.sunlight_window.pop(0)

        # Calculate current slope
        current_slope = self.calculate_slope(self.sunlight_window)
        print("current_slope", current_slope)

        # Determine conditions
        above_threshold = current_sunlight > self.sunlight_threshold
        current_trend = self.last_trend  # Keep previous trend unless we see a real change

        if len(self.sunlight_window) >= 2:
            if self.sunlight_window[-1] < self.sunlight_window[-2]:
                current_trend = "DECREASING"
            elif self.sunlight_window[-1] > self.sunlight_window[-2]:
                current_trend = "INCREASING"
            # If equal, keep previous trend

            # Only trade if we have a confirmed trend
            if current_trend == "DECREASING" and not above_threshold and self.position < self.position_limit:
                # Go max long when sunlight is decreasing while above threshold
                if normal_best_ask != float('inf'):
                    buy_quantity = min(
                        self.position_limit - self.position,  # Go to max position
                        abs(order_depth.sell_orders[normal_best_ask])
                    )

                    if buy_quantity > 0:
                        self.position += buy_quantity
                        orders["MAGNIFICENT_MACARONS"].append(
                            Order("MAGNIFICENT_MACARONS", normal_best_ask, buy_quantity)
                        )

            elif current_trend == "INCREASING" and current_slope > self.slope_threshold and self.position > 0:
                # Only close long position when sunlight is increasing with significant slope
                if normal_best_bid != 0:
                    sell_quantity = min(
                        self.position,  # Close entire position
                        abs(order_depth.buy_orders[normal_best_bid])
                    )

                    if sell_quantity > 0:
                        self.position -= sell_quantity
                        orders["MAGNIFICENT_MACARONS"].append(
                            Order("MAGNIFICENT_MACARONS", normal_best_bid, -sell_quantity)
                        )

        self.last_trend = current_trend

        # 1. RAINFOREST_RESIN (Kai’s version)
        if Product.RAINFOREST_RESIN in state.order_depths:
            od = state.order_depths[Product.RAINFOREST_RESIN]
            # width
            if od.sell_orders and od.buy_orders:
                best_ask = min(od.sell_orders)
                best_bid = max(od.buy_orders)
                width = (best_ask - best_bid) / 2
            else:
                width = 5
            # fair
            tot_v = sum(p * v for p, v in od.buy_orders.items()) \
                    + sum(p * abs(v) for p, v in od.sell_orders.items())
            tot_vol = sum(od.buy_orders.values()) + sum(abs(v) for v in od.sell_orders.values())
            fair = tot_v / tot_vol if tot_vol > 0 else 10000
            pos = state.position.get(Product.RAINFOREST_RESIN, 0)
            limit = self.LIMIT[Product.RAINFOREST_RESIN]
            result[Product.RAINFOREST_RESIN] = self.rainforest_resin_orders(
                od, fair, width, pos, limit
            )
        else:
            logger.print("RAINFOREST_RESIN not found")

        # 2. KELP (Kai’s Trade.kelp)
        if Product.KELP in state.order_depths:
            kelp_status = Status(Product.KELP, state)
            result[Product.KELP] = Trade.kelp(kelp_status)
        else:
            logger.print("KELP not found")

        # 3. SQUID_INK (Kai’s take_squid_orders)
        if Product.SQUID_INK in state.order_depths:
            self.adjust_desired_positions(state, Product.SQUID_INK)
            asset = Product.SQUID_INK
            position = 0
            try:
                position = state.position[asset]
            except:
                position = 0
            target_position = int(self.desired_position_pct[Product.SQUID_INK] * self.LIMIT[asset])
            position_difference = target_position - position
            if True:#not target_position == 0:
                order_depth = state.order_depths[asset]
                # Get market prices
                result[asset] = []
                best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
                if position_difference > 0:
                    # Need to buy
                    if best_ask != float('inf'):
                        ask_quantity = order_depth.sell_orders.get(best_ask, 0)
                        buy_quantity = min(
                            position_difference,
                            abs(ask_quantity)
                        )

                        if buy_quantity > 0:
                            result[asset].append(
                                Order(asset, best_ask, buy_quantity)
                            )

                elif position_difference < 0:
                    # Need to sell
                    if best_bid != 0:
                        bid_quantity = order_depth.buy_orders.get(best_bid, 0)
                        sell_quantity = min(
                            abs(position_difference),
                            abs(bid_quantity)
                        )

                        if sell_quantity > 0:
                            result[asset].append(
                                Order(asset, best_bid, -sell_quantity)
                            )
            else:
                pos = state.position.get(Product.SQUID_INK, 0)
                fair, vwap = self.squid_ink_fair_value(
                    state.order_depths[Product.SQUID_INK], traderObject
                )
                orders, _, _ = self.take_squid_orders(
                    Product.SQUID_INK,
                    vwap + target_position,
                    state.order_depths[Product.SQUID_INK],
                    self.params[Product.SQUID_INK]["take_width"],
                    pos,
                    self.params[Product.SQUID_INK]["prevent_adverse"],
                    self.params[Product.SQUID_INK]["adverse_volume"],
                    state.order_depths.get(Product.KELP),
                    26,
                    0.3,
                )
                result[Product.SQUID_INK] = orders
        else:
            logger.print("SQUID_INK not found")

        # 4. Basket spreads (unchanged)
        if Product.CROISSANTS in state.order_depths:
            self.adjust_desired_positions(state, Product.CROISSANTS)
            asset = Product.CROISSANTS
        croissant_target_position = int(self.desired_position_pct[Product.CROISSANTS] * self.LIMIT[Product.CROISSANTS])
        if croissant_target_position == 0:
            for basket, weights in [
                (Product.PICNIC_BASKET1, BASKET1_WEIGHTS),
                (Product.PICNIC_BASKET2, BASKET2_WEIGHTS),
            ]:
                spread_orders = self.spread_orders(
                    order_depths=state.order_depths,
                    product=basket,
                    basket_weights=weights,
                    basket_position=state.position.get(basket, 0),
                    spread_data=traderObject.setdefault(basket, {"spread_history": [], "prev_zscore": 0}),
                )
                if spread_orders:
                    for prod, orders in spread_orders.items():
                        result.setdefault(prod, []).extend(orders)
        else:
            assets = [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2, Product.CROISSANTS]
            for asset in assets:
                if asset not in state.order_depths:
                    continue
                position = 0
                try:
                    position = state.position[asset]
                except:
                    position = 0
                target_position = int(self.desired_position_pct[Product.CROISSANTS] * self.LIMIT[asset])
                position_difference = target_position - position
                if True:#not target_position == 0:
                    order_depth = state.order_depths[asset]
                    # Get market prices
                    result[asset] = []
                    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
                    if position_difference > 0:
                        # Need to buy
                        if best_ask != float('inf'):
                            ask_quantity = order_depth.sell_orders.get(best_ask, 0)
                            buy_quantity = min(
                                position_difference,
                                abs(ask_quantity)
                            )

                            if buy_quantity > 0:
                                result[asset].append(
                                    Order(asset, best_ask, buy_quantity)
                                )

                    elif position_difference < 0:
                        # Need to sell
                        if best_bid != 0:
                            bid_quantity = order_depth.buy_orders.get(best_bid, 0)
                            sell_quantity = min(
                                abs(position_difference),
                                abs(bid_quantity)
                            )

                            if sell_quantity > 0:
                                result[asset].append(
                                    Order(asset, best_bid, -sell_quantity)
                                )
        
        # 5. VOLCANIC_ROCK + vouchers
        if Product.VOLCANIC_ROCK in state.order_depths:
            # underlying
            self.adjust_desired_positions(state, Product.VOLCANIC_ROCK)

            #result[Product.VOLCANIC_ROCK] = Trade.volcanic_rock_copy(state)
            if True:
                assets = [Product.VOLCANIC_ROCK, Product.VOLCANIC_ROCK_VOUCHER_9500]
                for asset in assets:
                    if asset not in state.order_depths:
                        continue
                    v_order_depth = state.order_depths[Product.VOLCANIC_ROCK]
                        # Get market prices
                    v_best_bid = max(v_order_depth.buy_orders.keys()) if v_order_depth.buy_orders else 0
                    v_best_ask = min(v_order_depth.sell_orders.keys()) if v_order_depth.sell_orders else float('inf')
                    if asset == Product.VOLCANIC_ROCK_VOUCHER_9500 and v_best_ask < 10000: 
                        continue
                    position = 0
                    try:
                        position = state.position[asset]
                    except:
                        position = 0
                    target_position = int(self.desired_position_pct[Product.VOLCANIC_ROCK] * self.LIMIT[asset])
                    position_difference = target_position - position
                    if True:#not target_position == 0:
                        result[asset] = []
                        order_depth = state.order_depths[asset]
                        # Get market prices
                        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else 0
                        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else float('inf')
                        if position_difference > 0:
                            # Need to buy
                            if best_ask != float('inf'):
                                ask_quantity = order_depth.sell_orders.get(best_ask, 0)
                                buy_quantity = min(
                                    position_difference,
                                    abs(ask_quantity)
                                )

                                if buy_quantity > 0:
                                    result[asset].append(
                                        Order(asset, best_ask, buy_quantity)
                                    )

                        elif position_difference < 0:
                            # Need to sell
                            if best_bid != 0:
                                bid_quantity = order_depth.buy_orders.get(best_bid, 0)
                                sell_quantity = min(
                                    abs(position_difference),
                                    abs(bid_quantity)
                                )

                                if sell_quantity > 0:
                                    result[asset].append(
                                        Order(asset, best_bid, -sell_quantity)
                                    )
            underlying = Status(Product.VOLCANIC_ROCK, state)



            # vouchers for 9750, 10000
            for strike in (9750, 10000):
                key = f"{Product.VOLCANIC_ROCK}_VOUCHER_{strike}"
                if key in state.order_depths:
                    st = Status(key, state, strike)
                    result[key] = Trade.voucher(st, underlying, strike)
        else:
            logger.print("VOLCANIC_ROCK not found")

        # 6. Update profit history (for Kai’s trend logic)
        for product, position in state.position.items():
            if product in state.order_depths and position != 0:
                # extract strike if voucher
                strike = int(product.rsplit("_", 1)[-1]) if "VOUCHER" in product else None
                ps = Status(product, state, strike)
                profit = position * (ps.vwap - 10000) / 100
                ps.update_profit_history(profit)

        # 7. Flush & return
        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData