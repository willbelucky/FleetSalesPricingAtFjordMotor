# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 30.
"""
import pandas as pd

# Set the PYTHONPATH as root folder, like PYTHONPATH=/Users/willbe/PycharmProjects/FleetSalesPricingAtFjordMotor
DATA_DIR = 'data/fleet_sales_pricing_at_fjord_motor.csv'


def get_all_data():
    """
    Get data of all bids.

    :return data: (DataFrame) 8 columns * 4000 rows
        columns unit_number         | (int) The number of vehicles.
                unit_price          | (int) The price of a vehicle.
                total_price         | (int) unit_number * unit_price
                win                 | (int) If Fjord win a fleet bid, win is 1. Else, win is 0.
                discount_rate       | (float) ( MSRP(the_manufacturer's_suggested_retail_price) - unit_price ) / MSRP
                unit_margin         | (int) MSRP - cost
                unit_sold_number    | (int) unit_number * win
                total_margin        | (int) unit_margin * unit_sold_number
    """
    data = pd.read_csv(DATA_DIR)
    return data


def get_police_data():
    """
    The bids 1 through 2000 were to police departments.

    :return data: (DataFrame) 8 columns * 2000 rows
        columns unit_number         | (int) The number of vehicles.
                unit_price          | (int) The price of a vehicle.
                total_price         | (int) unit_number * unit_price
                win                 | (int) If Fjord win a fleet bid, win is 1. Else, win is 0.
                discount_rate       | (float) ( MSRP(the_manufacturer's_suggested_retail_price) - unit_price ) / MSRP
                unit_margin         | (int) MSRP - cost
                unit_sold_number    | (int) unit_number * win
                total_margin        | (int) unit_margin * unit_sold_number
    """
    data = get_all_data()
    data = data.iloc[:2000, :]
    return data


def get_corporate_buyer_data():
    """
    The bids 2001 through 4000 were to corporate buyers.

    :return data: (DataFrame) 8 columns * 2000 rows
        columns unit_number         | (int) The number of vehicles.
                unit_price          | (int) The price of a vehicle.
                total_price         | (int) unit_number * unit_price
                win                 | (int) If Fjord win a fleet bid, win is 1. Else, win is 0.
                discount_rate       | (float) ( MSRP(the_manufacturer's_suggested_retail_price) - unit_price ) / MSRP
                unit_margin         | (int) MSRP - cost
                unit_sold_number    | (int) unit_number * win
                total_margin        | (int) unit_margin * unit_sold_number
    """
    data = get_all_data()
    data = data.iloc[2000:, :]
    return data


if __name__ == '__main__':
    print(get_police_data())
    print('-' * 70)
    print(get_corporate_buyer_data())
