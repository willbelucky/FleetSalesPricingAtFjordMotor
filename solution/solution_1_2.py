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
MSRP = 25000

# Make an initial parameter guess as average of cost and MSRP.
INITIAL_GUESS = np.array([(COST + MSRP) / 2])


def probability_of_winning_to_bid_model(intercept, beta_1, unit_prices):
    return 1 / (1 + np.exp(intercept + beta_1 * unit_prices))


def negative_margin_model(unit_price, *args):
    intercept = args[0]
    beta_1 = args[1]
    probability = probability_of_winning_to_bid_model(intercept, beta_1, unit_prices=unit_price / MSRP)
    unit_margin = unit_price - COST
    return -probability * unit_margin


def negative_log_likelihood(params, *args):
    intercept = params[0]
    beta_1 = params[1]

    y_observations = np.array(args[0])
    unit_prices = np.array(args[1])

    # Calculate the predicted values from the initial parameter guesses
    y_predictions = probability_of_winning_to_bid_model(intercept, beta_1, unit_prices)

    # Calculate the log-likelihood.
    log_likelihood = np.sum(stats.norm.logpdf(y_observations, loc=y_predictions))

    # Return the negative log likelihood.
    return -log_likelihood


def maximize_log_likelihood(y, x):
    # Make a list of initial parameter guesses (intercept, beta_1)
    initial_guess = np.array([1, 0])

    # Minimize a negative log likelihood for maximizing a log likelihood.
    results = minimize(negative_log_likelihood, x0=initial_guess, args=(y, x))

    intercept = results.x[0]
    beta_1 = results.x[1]

    return intercept, beta_1


def get_data_y_x(get_data_function):
    data = get_data_function()
    # Set up your observed y values.
    y_observation = data[data_column.WIN].values
    # Set up your x values.
    x = (data[data_column.UNIT_PRICE] / MSRP).values

    return data, y_observation, x


def get_optimal_price(intercept, beta_1):
    # Minimize a negative margin for maximizing a margin.
    optimal_price = minimize(negative_margin_model, INITIAL_GUESS, args=(intercept, beta_1)).x[0]
    return optimal_price


if __name__ == '__main__':
    # 1_A
    print('1_A. What are the value of a and b that maximize the sum of log likelihood?')
    all_data, all_y_observation, all_x = get_data_y_x(get_all_data)
    all_intercept, all_beta_1 = maximize_log_likelihood(all_y_observation, all_x)
    print('a:{:.3f}, b:{:.3f}'.format(all_intercept, all_beta_1))
    print('-' * 70)

    # 1_B
    print('1_B. What is the optimum price Fjord should offer, '
          'assuming it is going to offer a single price for each bid?')
    all_optimal_price = get_optimal_price(all_intercept, all_beta_1)
    print(all_optimal_price)
    print('-' * 70)

    # 1_C
    print('1_C. What would the expected total contribution have been for the 4,000 bids?')
    all_optimal_total_contribution = -negative_margin_model(all_optimal_price, all_intercept, all_beta_1) * all_data[
        data_column.UNIT_NUMBER].sum()
    print(all_optimal_total_contribution)
    print('-' * 70)

    # 1_D
    print('1_D. How does this compare to the contribution that Fjord actually received?')
    actual_total_contribution = all_data[data_column.TOTAL_MARGIN].sum()
    print('The optimal total contribution is {:.3f} times the actual total contribution.'.format(
        all_optimal_total_contribution / actual_total_contribution))
    print('-' * 70)

    # 2_A
    print('2_A. What are the corresponding values of a and b for each?')
    print('The police')
    police_data, police_y_observation, police_x = get_data_y_x(get_police_data)
    police_intercept, police_beta_1 = maximize_log_likelihood(police_y_observation, police_x)
    print('a:{:.3f}, b:{:.3f}'.format(police_intercept, police_beta_1))

    print('Corporate buyers')
    corporate_buyer_data, corporate_buyer_y_observation, corporate_buyer_x = get_data_y_x(
        get_corporate_buyer_data)
    corporate_buyer_intercept, corporate_buyer_beta_1 = maximize_log_likelihood(
        corporate_buyer_y_observation,
        corporate_buyer_x)
    print('a:{:.3f}, b:{:.3f}'.format(corporate_buyer_intercept, corporate_buyer_beta_1))
    print('-' * 70)

    # 2_B
    print('2_B. What are the optimum prices Fjord should offer to the police?')
    police_optimal_price = get_optimal_price(police_intercept, police_beta_1)
    print(police_optimal_price)
    print('-' * 70)

    # 2_C
    print('2_C. What are the optimum prices Fjord should offer to corporate buyer?')
    corporate_buyer_optimal_price = get_optimal_price(corporate_buyer_intercept, corporate_buyer_beta_1)
    print(corporate_buyer_optimal_price)
    print('-' * 70)

    # 2_D
    print('2_D. What would the expected contribution have been '
          'if Fjord had used the prices in the 4,000 bids in the database?')
    police_optimal_total_contribution = -negative_margin_model(
        police_optimal_price, police_intercept, police_beta_1
    ) * police_data[data_column.UNIT_NUMBER].sum()
    corporate_buyer_optimal_total_contribution = -negative_margin_model(
        corporate_buyer_optimal_price, corporate_buyer_intercept, corporate_buyer_beta_1
    ) * corporate_buyer_data[data_column.UNIT_NUMBER].sum()
    optimal_total_contribution_sum = police_optimal_total_contribution + corporate_buyer_optimal_total_contribution
    print(optimal_total_contribution_sum)
    print('-' * 70)

    # 2_E
    print('2_E. What is the difference between the contribution actually received and the best '
          'that Fjord could do when it could not differentiate between the police and corporate buyers?')
    print('The sum of the police and corporate buyer optimal total contribution is {:.3f} times the actual total '
          'contribution.'.format(optimal_total_contribution_sum / actual_total_contribution))
    print('-' * 70)
