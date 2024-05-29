import time
from typing import Generator, List, TypeVar

try:
    import dask
    from distributed import Client, as_completed
except ImportError:
    pass

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import Algorithm, DynamicAlgorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import DynamicProblem, Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator, DominanceComparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import (
    RankingAndDensityEstimatorReplacement,
    RemovalPolicyType,
)
from jmetal.util.termination_criterion import TerminationCriterion

from surrogate_models.regressor_chain_surrogate import RegressorChainSurrogate
from surrogate_models.surrogate import Surrogate

from jmetal.logger import get_logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S = TypeVar("S")
R = TypeVar("R")

logger = get_logger(__name__)

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class S_NSGAII(GeneticAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(
            MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])
        ),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
        batch_sample_percentaje: float = 0.1,
        surrogate_ml: Surrogate = None,
    ):
        """
        NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        """
        super(S_NSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
        )
        self.dominance_comparator = dominance_comparator
        self.batch_sample_percentaje = batch_sample_percentaje
        self.surrogate_ml = surrogate_ml
        self.identification = id(self)
        self.divertsity_values = None

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "NSGAII"
    
    def get_diversity(self):
        self.divertsity_values = self.surrogate_ml.get_diversity().copy()
        return self.divertsity_values
    
    def run(self):
        """Execute the algorithm."""
        '''Create the initial population and model'''
        evaluate_surrogate = False
        train_surrogate = False
        historical_solutions = []
        train_cycle = 0
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        '''Add the initial population to the historical solutions for train data'''
        self.init_progress()
        for solution in self.solutions:
            historical_solutions.append(solution)
        
        while not self.stopping_condition_is_met():
            mating_population = self.selection(self.solutions)
            offspring_population = self.reproduction(mating_population)
            
            if  (self.get_termination_criterion().current_evaluation() % int(self.batch_sample_percentaje*self.get_termination_criterion().max_evaluation()) == 0):
                evaluate_surrogate = not evaluate_surrogate
                train_surrogate = False

            if evaluate_surrogate:
                if not train_surrogate:
                    train_cycle += 1
                    logger.debug("Train cycle: ",train_cycle)
                    self.surrogate_ml.fit(historical_solutions)
                    offspring_population = self.surrogate_ml.evaluate(offspring_population)
                    train_surrogate = True
                else:
                    offspring_population = self.surrogate_ml.evaluate(offspring_population)

            else:
                offspring_population = self.evaluate(offspring_population)
            
            '''Replacement of the solutions in the population'''
            self.solutions = self.replacement(self.solutions, offspring_population)

            '''Add the new solutions to the historical solutions for train data'''
            if not evaluate_surrogate:
                for solution in self.solutions:
                    historical_solutions.append(solution)

            self.update_progress()
        
        logger.debug("Number of surrogate evaluation: ",self.surrogate_ml.internal_execution)
        logger.debug("Total surrogate evaluation: ", self.population_size*self.surrogate_ml.internal_execution)
        logger.debug("Total evaluation: ", self.get_termination_criterion().max_evaluation())

        self.total_computing_time = time.time() - self.start_computing_time


class NSGAII(GeneticAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(
            MultiComparator([FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])
        ),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
        batch_sample_percentaje: float = 0.1,
    ):
        """
        NSGA-II implementation as described in

        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017

        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).

        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1.

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        """
        super(NSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
        )
        self.dominance_comparator = dominance_comparator
        self.batch_sample_percentaje = batch_sample_percentaje
        self.previous_data = None
        '''diversity: float:total, float:valid'''
        self.diversity = list()
        self.previous_len_data=0
        self.len_data = 0
        self.len_entry_data = 0

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.

        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "NSGAII"
    
    def step(self):
        mating_population = self.selection(self.solutions)
        offspring_population = self.reproduction(mating_population)
        offspring_population = self.evaluate(offspring_population)

        self.solutions = self.replacement(self.solutions, offspring_population)
    
    def run(self):
        historical_solutions = []
        
        add_data = True
        evaluate_diversity = False

        """Execute the algorithm."""
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()

        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        for solution in self.solutions:
           historical_solutions.append(solution.variables + solution.objectives)

        while not self.stopping_condition_is_met():
            if  (self.get_termination_criterion().current_evaluation() % int(self.batch_sample_percentaje*self.get_termination_criterion().max_evaluation()) == 0):
                add_data = not add_data
                if not add_data:
                    evaluate_diversity = True
            

            self.step()

            if add_data:
                for solution in self.solutions:
                    historical_solutions.append(solution.variables + solution.objectives)
            
            if evaluate_diversity:
                evaluate_diversity = False
                self.valid_data(historical_solutions)

            self.update_progress()

        self.total_computing_time = time.time() - self.start_computing_time
    
    def valid_data(self, data):
        '''Check total amount of new data'''
        if self.len_data == 0:
            self.len_data = len(data)
        else:
            self.previous_len_data = self.len_data
            self.len_data = len(data)
        
        self.len_entry_data = self.len_data - self.previous_len_data

        
        '''Add the actual data to previous data for not repeat rows'''
        complete_data = pd.DataFrame(data)
        no_duplicates_data = complete_data.drop_duplicates()
        if self.previous_data is not None:
            valid_data = no_duplicates_data.merge(self.previous_data, how='left', indicator=True)
            valid_data = valid_data[valid_data['_merge'] == 'left_only'].drop(columns='_merge')            
        else:
            valid_data = no_duplicates_data
        
        if self.previous_data is None:
            self.previous_data = valid_data
        else:
            self.previous_data = pd.concat([self.previous_data, valid_data])
        
        self.diversity.append([self.len_entry_data, valid_data.shape[0]])

    def get_diversity(self):
        return self.diversity
    

def reproduction(mating_population: List[S], problem, crossover_operator, mutation_operator) -> S:
    offspring_pool = []
    for parents in zip(*[iter(mating_population)] * 2):
        offspring_pool.append(crossover_operator.execute(parents))

    offspring_population = []
    for pair in offspring_pool:
        for solution in pair:
            mutated_solution = mutation_operator.execute(solution)
            offspring_population.append(mutated_solution)

    return problem.evaluate(offspring_population[0])
