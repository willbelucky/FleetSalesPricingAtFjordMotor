# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2018. 1. 30.
"""
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize

from data import data_column
from data.data_reader import get_all_data, get_police_data, get_corporate_buyer_data

# Setting data
COST = 15000
DIFFERENT_COST = 16000
MSRP = 25000

# Make an initial parameter guess as average of cost and MSRP.
INITIAL_PRICE_GUESS = np.array([20000])


def probability_of_winning_to_bid_model(intercept, beta_1, beta_2, unit_prices, unit_numbers):
    return 1 / (1 + np.exp(intercept + beta_1 * unit_prices + beta_2 * unit_numbers))


def negative_margin_model(unit_price, *args):
    different_cost = args[0]
    intercept = args[1]
    beta_1 = args[2]
    beta_2 = args[3]
    unit_number = args[4]
    probability = probability_of_winning_to_bid_model(intercept, beta_1, beta_2, unit_prices=unit_price / MSRP,
                                                      unit_numbers=unit_number)
    if different_cost:
        unit_margin = unit_price - DIFFERENT_COST
    else:
        unit_margin = unit_price - COST
    return -probability * unit_margin


def negative_log_likelihood(params, *args):
    intercept = params[0]
    beta_1 = params[1]
    beta_2 = params[2]

    y_observations = np.array(args[0])
    unit_prices = np.array(args[1][:, 0])
    unit_numbers = np.array(args[1][:, 1])

    # Calculate the predicted values from the initial parameter guesses
    y_predictions = probability_of_winning_to_bid_model(intercept, beta_1, beta_2, unit_prices, unit_numbers)

    # Calculate the log-likelihood.
    log_likelihood = np.sum(stats.norm.logpdf(y_observations, loc=y_predictions))

    # Return the negative log likelihood.
    return -log_likelihood


def maximize_log_likelihood(y, x, different_cost=False):
    # Make a list of initial parameter guesses (intercept, beta_1)
    initial_guess = np.array([different_cost, 1, 0, 0])

    # Minimize a negative log likelihood for maximizing a log likelihood.
    results = minimize(negative_log_likelihood, x0=initial_guess, args=(y, x))

    intercept = results.x[0]
    beta_1 = results.x[1]
    beta_2 = results.x[2]

    return intercept, beta_1, beta_2


def get_data_y_x(get_data_function):
    data = get_data_function()
    # Set up your observed y values.
    y_observation = data[data_column.WIN].values
    # Set up your x values.
    data[data_column.UNIT_PRICE] = data[data_column.UNIT_PRICE] / MSRP
    x = (data[[data_column.UNIT_PRICE, data_column.UNIT_NUMBER]]).values

    return data, y_observation, x


def get_optimal_price(intercept, beta_1, beta_2, unit_number):
    # Minimize a negative margin for maximizing a margin.
    optimal_price = \
        minimize(negative_margin_model, INITIAL_PRICE_GUESS, args=(False, intercept, beta_1, beta_2, unit_number)).x[0]
    return optimal_price


if __name__ == '__main__':
    # 3_A
    print('What is the resulting improvement in total log likelihood?')
    all_data, all_y_observation, all_x = get_data_y_x(get_all_data)
    average_unit_number = all_data[data_column.UNIT_NUMBER].mean()
    all_intercept, all_beta_1, all_beta_2 = maximize_log_likelihood(all_y_observation, all_x)
    all_optimal_price = get_optimal_price(all_intercept, all_beta_1, all_beta_2, average_unit_number)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(all_intercept, all_beta_1, all_beta_2,
                                                                  all_optimal_price))
    print('-' * 70)

    # 3_B
    print('How does this compare with the improvement from differentiating police and corporate sales?')
    print('The police')
    police_data, police_y_observation, police_x = get_data_y_x(get_police_data)
    police_intercept, police_beta_1, police_beta_2 = maximize_log_likelihood(
        police_y_observation,
        police_x)
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1, police_beta_2, average_unit_number)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        police_intercept,
        police_beta_1,
        police_beta_2,
        police_optimal_price
    ))

    print('Corporate buyers')
    corporate_buyer_data, corporate_buyer_y_observation, corporate_buyer_x = get_data_y_x(
        get_corporate_buyer_data)
    corporate_buyer_intercept, corporate_buyer_beta_1, corporate_buyer_beta_2 = maximize_log_likelihood(
        corporate_buyer_y_observation,
        corporate_buyer_x)
    corporate_buyer_optimal_price = get_optimal_price(corporate_buyer_intercept, corporate_buyer_beta_1,
                                                      corporate_buyer_beta_2, average_unit_number)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        corporate_buyer_intercept,
        corporate_buyer_beta_1,
        corporate_buyer_beta_2,
        corporate_buyer_optimal_price
    ))

    police_optimal_total_contribution = -negative_margin_model(
        False,
        police_optimal_price,
        police_intercept,
        police_beta_1,
        police_beta_2,
        average_unit_number
    ) * police_data[data_column.UNIT_NUMBER].sum()

    corporate_buyer_optimal_total_contribution = -negative_margin_model(
        False,
        corporate_buyer_optimal_price, corporate_buyer_intercept,
        corporate_buyer_beta_1,
        corporate_buyer_beta_2,
        average_unit_number
    ) * corporate_buyer_data[data_column.UNIT_NUMBER].sum()

    optimal_total_contribution_sum = police_optimal_total_contribution + corporate_buyer_optimal_total_contribution
    print(optimal_total_contribution_sum)
    print('-' * 70)

    # 3_C
    print('What are the optimal prices Fjord should charge for orders of 20 cars '
          'and for orders of 40 cars to police departments and to corporate purchasers, respectively?')
    print('The police for orders of 20 cars')
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1, police_beta_2, 20)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        police_intercept,
        police_beta_1,
        police_beta_2,
        police_optimal_price
    ))
    print('The police for orders of 40 cars')
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1, police_beta_2, 40)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        police_intercept,
        police_beta_1,
        police_beta_2,
        police_optimal_price
    ))
    print('Corporate buyers for orders of 20 cars')
    corporate_buyer_optimal_price = get_optimal_price(corporate_buyer_intercept, corporate_buyer_beta_1,
                                                      corporate_buyer_beta_2, 20)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        corporate_buyer_intercept,
        corporate_buyer_beta_1,
        corporate_buyer_beta_2,
        corporate_buyer_optimal_price
    ))
    print('Corporate buyers for orders of 40 cars')
    corporate_buyer_optimal_price = get_optimal_price(corporate_buyer_intercept, corporate_buyer_beta_1,
                                                      corporate_buyer_beta_2, 40)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        corporate_buyer_intercept,
        corporate_buyer_beta_1,
        corporate_buyer_beta_2,
        corporate_buyer_optimal_price
    ))
    print('-' * 70)

    # 4_A
    print('How would this change the optimal price charged to police departments for 20 vehicles?')
    police_intercept, police_beta_1, police_beta_2 = maximize_log_likelihood(
        police_y_observation,
        police_x,
        False
    )
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1, police_beta_2, 20)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        police_intercept,
        police_beta_1,
        police_beta_2,
        police_optimal_price
    ))
    print('The optimal price is not changed.')
    print('-' * 70)

    # 4_B
    print('How would this change the optimal price charged to police departments for 40 vehicles?')
    police_intercept, police_beta_1, police_beta_2 = maximize_log_likelihood(
        police_y_observation,
        police_x,
        False
    )
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1, police_beta_2, 40)
    print('a:{:.3f}, b:{:.3f}, c:{:.3f}, optimal_price:{}'.format(
        police_intercept,
        police_beta_1,
        police_beta_2,
        police_optimal_price
    ))
    print('The optimal price is not changed.')
    print('-' * 70)
