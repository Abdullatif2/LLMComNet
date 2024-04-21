import numpy as np
from typing import List
from langchain.pydantic_v1 import BaseModel, Field, validator


class SettingRecommendation(BaseModel):
    interest: str = Field(description="User's question regarding a setting need.")
    recommended_setting: str = Field(description="Recommended setting response.")
    
    @validator('interest')
    def validate_interest(cls, value):
        if not value:
            raise ValueError("Interest cannot be empty.")
        return value

class Summation(BaseModel):
    number1: float = Field(description="First number for summation.")
    number2: float = Field(description="Second number for summation.")
    
    @validator("number1", "number2")
    def validate_numbers(cls, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Inputs must be numbers.")
        return value


def simulate_channel(n):
    return np.random.uniform(0.5, 1.5, size=n)


class BandwidthOptimization(BaseModel):
    """Solve the bandwidth allocation with given channel state information and number of users as well as the total bandwidth."""
    num_users: int = Field(description="Number of users in the system.")
    total_bandwidth: float = Field(description="Total available bandwidth.")
    energy_cost_coefficient: float = Field(description="Energy cost coefficient.")
    channel_gains: List[float] = Field(description="Channel gains for each user.")

    class Config:
            num_users_example = 10
            channel_gains_example = np.random.uniform(0.5, 1.5, num_users_example).tolist()
            schema_extra = {
                "example": {
                    "num_users": num_users_example,
                    "total_bandwidth": 100,
                    "energy_cost_coefficient": 1,
                    "channel_gains": channel_gains_example,
                }
            }
    @validator("channel_gains")
    def gains_must_be_positive(cls, value):
        assert all(g > 0 for g in value), "Channel gains must be positive."
        return value

class BeamPrediction(BaseModel):
    """Predict optimal beamforming vectors based on user locations and channel state information."""
    num_antennas: int = Field(description="Number of antennas at the base station.")
    user_positions: List[List[float]] = Field(description="Coordinates of users in the system.")

    class Config:
        schema_extra = {
            "example": {
                "num_antennas": 8,
                "user_positions": [[10, 5], [15, 20], [30, 25]],
            }
        }

    @validator("user_positions")
    def positions_must_be_valid(cls, value):
        assert all(len(pos) == 2 for pos in value), "Each position must be a 2D coordinate."
        return value
    
class ChannelEstimation(BaseModel):
    """Estimate the channel conditions accurately and  Estimate the channel conditions for communications between base station and users."""
    num_users: int = Field(description="Number of users in the system.")
    estimation_method: str = Field(description="Method used for estimating the channel.")

    class Config:
        schema_extra = {
            "example": {
                "num_users": 10,
                "estimation_method": "Least Squares",
            }
        }

class PowerAllocation(BaseModel):
    """Optimize power allocation to minimize interference and maximize throughput."""
    num_channels: int = Field(description="Number of channels available for transmission.")
    power_limits: List[float] = Field(description="Maximum power limit for each channel in watts.")

    class Config:
        schema_extra = {
            "example": {
                "num_channels": 5,
                "power_limits": [20.0, 20.0, 15.0, 15.0, 10.0],
            }
        }

    @validator("power_limits")
    def limits_must_be_positive(cls, values):
        assert all(p >= 0 for p in values), "Power limits must be non-negative."
        return values