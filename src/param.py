import random
import time
import math

from log import Log

logger = Log('Param')


class Param:
    def __init__(
        self,
        default_value,
        direction: str,
        low_limit: float,
        high_limit: float,
        id: int,
        name: str,
        starting_multiplier: float = 0.0,
        percentage_drop: float = 0.0,
        current_value=None,
        starting_value_if_zero=None,
    ):
        if not current_value:
            self.current_value = default_value
        else:
            self.current_value = current_value

        assert direction == "positive" or direction == "negative" \
            'direction should be either positive or negative: {}'.format(direction)

        self.default_value = default_value
        self.direction = direction
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.starting_multiplier = starting_multiplier
        self.id = id
        self.name = name
        self.percentage_drop = percentage_drop

        self.mutated = False
        self.modified_at = None
        self.starting_value_if_zero = starting_value_if_zero

        # override direction
        self.direction = 'positive' if self.default_value == self.low_limit else 'negative'

    def __str__(self):
        return (
            "{default_value:"
            + str(self.default_value)
            + ",current_value:"
            + str(self.current_value)
            + ",direction:"
            + str(self.direction)
            + ",low_limit:"
            + str(self.low_limit)
            + ",high_limit:"
            + str(self.high_limit)
            + ",starting_multiplier:"
            + str(self.starting_multiplier)
            + ",id:"
            + str(self.id)
            + ",name:"
            + str(self.name)
            + "}"
        )

    def get_default_value(self, non_zero_value: bool = False) -> float:
        if non_zero_value and self.default_value == 0.0:
            return self.starting_value_if_zero
        return self.default_value

    def get_starting_value_if_zero(self) -> float:
        return self.starting_value_if_zero

    def get_id(self) -> int:
        return self.id

    def is_mutated(self) -> bool:
        return self.mutated

    def get_modified_at(self) -> float:
        return self.modified_at

    def get_current_value(self) -> float:
        return self.current_value

    def set_starting_multiplier(self, starting_multiplier: float):
        assert starting_multiplier > 0.0, 'starting_multiplier > 0: {}'.format(starting_multiplier)
        self.starting_multiplier = starting_multiplier

    def get_starting_multiplier(self) -> float:
        return self.starting_multiplier

    def set_percentage_drop(self, percentage_drop: float):
        assert percentage_drop > 0.0, 'percentage_drop > 0.0: {}'.format(percentage_drop)
        self.percentage_drop = percentage_drop

    def get_percentage_drop(self) -> float:
        return self.percentage_drop

    def get_direction(self) -> str:
        return self.direction

    def get_low_limit(self) -> float:
        return self.low_limit

    def set_low_limit(self, low_limit: float) -> None:
        self.low_limit = low_limit

    def get_high_limit(self) -> float:
        return self.high_limit

    def set_high_limit(self, high_limit: float) -> None:
        self.high_limit = high_limit

    def get_name(self) -> str:
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Param):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.id == other.id and self.current_value == other.current_value

    def mutate_random(self) -> bool:

        self.current_value = random.uniform(a=self.low_limit, b=self.high_limit)
        self.modified_at = time.time()
        return True

    def mutate_linear(self, direction_if_unknown: str = "positive") -> bool:

        mutated = False
        if self.current_value == 0.0:
            new_value = self.starting_value_if_zero
        else:
            new_value = self.current_value

        if self.direction == "positive" or (self.direction == "unknown" and direction_if_unknown == "positive"):
            new_value = self.current_value + self.default_value
        elif self.direction == "negative" or (self.direction == "unknown" and direction_if_unknown == "negative"):
            new_value = self.current_value - self.default_value

        if self.low_limit <= new_value <= self.high_limit:
            self.current_value = new_value
            mutated = True
        elif new_value < self.low_limit and self.current_value != self.low_limit:
            self.current_value = self.low_limit
            mutated = True
        elif new_value > self.high_limit and self.current_value != self.high_limit:
            self.current_value = self.high_limit
            mutated = True

        self.mutated = mutated
        if mutated:
            self.modified_at = time.time()
        return mutated

    def mutate(self, halve_or_double: str = None, ignore_starting_multiplier: bool = False) -> bool:

        if self.current_value == 0.0:
            new_value = self.starting_value_if_zero
        else:
            new_value = self.current_value

        if (new_value != self.default_value \
                and (new_value == self.low_limit or new_value == self.high_limit))\
                or (self.low_limit == 0.0 and math.isclose(new_value, 0.0, abs_tol=1.e-6)):
            logger.debug('current value reached limit: {}'.format(new_value))
            return False

        if self.starting_multiplier != 0.0 and not ignore_starting_multiplier and not halve_or_double:
            if self.direction == "positive":
                new_value = new_value * self.starting_multiplier
            elif self.direction == "negative":
                new_value = new_value * (-self.starting_multiplier)
            else:
                if random.random() <= 0.5:
                    new_value = new_value * self.starting_multiplier
                else:
                    new_value = new_value * (-self.starting_multiplier)
        else:
            # we do not know what the starting_multiplier is so the best guess is to double the param value or
            # to halve it
            if halve_or_double and (halve_or_double == "halve" or halve_or_double == "double"):
                # if it is known then act accordingly
                if self.direction == "positive":
                    new_value = new_value * 2.0 if halve_or_double == "double" else new_value * 0.5
                elif self.direction == "negative":
                    if new_value < 0:
                        # otherwise the new_value will be positive again
                        new_value = new_value * 2.0 if halve_or_double == "double" else new_value * 0.5
                    else:
                        new_value = new_value * (-2.0) if halve_or_double == "double" else new_value * (-0.5)
                else:
                    if random.random() <= 0.5:
                        new_value = new_value * 2.0 if halve_or_double == "double" else new_value * 0.5
                    else:
                        new_value = new_value * (-2.0) if halve_or_double == "double" else new_value * (-0.5)
            else:
                # otherwise choose randomly
                if random.random() <= 0.5:
                    # double
                    if self.direction == "positive":
                        new_value = new_value * 2.0
                    elif self.direction == "negative":
                        new_value = new_value * (-2.0)
                    else:
                        if random.random() <= 0.5:
                            new_value = new_value * 2.0
                        else:
                            new_value = new_value * (-2.0)
                else:
                    # halve
                    if self.direction == "positive":
                        new_value = new_value * 0.5
                    elif self.direction == "negative":
                        new_value = new_value * (-0.5)
                    else:
                        if random.random() <= 0.5:
                            new_value = new_value * 0.5
                        else:
                            new_value = new_value * (-0.5)

        mutated = self._adjust_within_limits(new_value=new_value)

        self.mutated = mutated
        if mutated:
            self.modified_at = time.time()
        return mutated

    def _adjust_within_limits(self, new_value: float) -> bool:
        if self.low_limit <= new_value <= self.high_limit:
            self.current_value = new_value
            return True

        if new_value < self.low_limit and self.current_value != self.low_limit:
            self.current_value = self.low_limit
            return True

        if new_value > self.high_limit and self.current_value != self.high_limit:
            self.current_value = self.high_limit
            return True

        return False
