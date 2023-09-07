import math
import numpy as np
import pandas as pd
import pygeohash as gh


def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    assert isinstance(X, pd.DataFrame)
    
    timedelta = (X["pickup_datetime"] - pd.Timestamp('2009-01-01T00:00:00', tz='UTC'))/pd.Timedelta(1,'D')
    
    pickup_dt = X["pickup_datetime"].dt.tz_convert("America/New_York").dt
    dow = pickup_dt.weekday
    hour = pickup_dt.hour
    month = pickup_dt.month

    hour_sin = np.sin(2 * math.pi / 24 * hour)
    hour_cos = np.cos(2 * math.pi / 24 * hour)
    
    return np.stack([hour_sin, hour_cos, dow, month, timedelta], axis=1)


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)
    lonlat_features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    
    def distances_vectorized(df: pd.DataFrame, start_lat: str, start_lon: str, end_lat: str, end_lon: str) -> dict:
        """
        Calculate the haversine and Manhattan distances between two points (specified in decimal degrees).
        Vectorized version for pandas df
        Computes distance in Km
        """       
        earth_radius = 6371

        lat_1_rad, lon_1_rad = np.radians(df[start_lat]), np.radians(df[start_lon])
        lat_2_rad, lon_2_rad = np.radians(df[end_lat]), np.radians(df[end_lon])

        dlon_rad = lon_2_rad - lon_1_rad
        dlat_rad = lat_2_rad - lat_1_rad

        manhattan_rad = np.abs(dlon_rad) + np.abs(dlat_rad)
        manhattan_km = manhattan_rad * earth_radius

        a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
        haversine_rad = 2 * np.arcsin(np.sqrt(a))
        haversine_km = haversine_rad * earth_radius

        return dict(
            haversine = haversine_km,
            manhattan = manhattan_km
        )
    
    res = distances_vectorized(X, *lonlat_features)

    return pd.DataFrame(res)
    

def compute_geohash(X: pd.DataFrame, precision: int = 5) -> np.ndarray:
    """
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon, lat) tuple, for pick-up, and drop-off
    """
    assert isinstance(X, pd.DataFrame)

    X["geohash_pickup"] = X.apply(lambda x: gh.encode(x.pickup_latitude, 
                                                      x.pickup_longitude, 
                                                      precision=precision),
                                    axis=1)
    
    X["geohash_dropoff"] = X.apply(lambda x: gh.encode(x.dropoff_latitude, 
                                                       x.dropoff_longitude, 
                                                       precision=precision),
                                    axis=1)

    return X[["geohash_pickup", "geohash_dropoff"]]
