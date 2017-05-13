# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:24:05 2017

@author: ACKWinDesk
"""
import random
from numpy.random import choice
import time
import copy
import cProfile
import pickle

# use information set because multiple histories lead to same game state
# Card points are negative, chip points are positive
class NoThanks(object):
    """Runs counter factual regret minimization training algorithm"""

    def __init__(self, game_parameters=None):
        self.game_tree = GameTree(game_parameters)

    def train(self, iterations=10):
        """Runs cfr iterations"""
        self.utils = [iterations*[0], iterations*[0]]
        NUM_PLAYERS = self.game_tree.game_parameters['NUM_PLAYERS']
        for k in range(iterations):
            for i in range(NUM_PLAYERS):
                self.utils[i][k] = self.cfr('', i, k, NUM_PLAYERS*[1])

    def play(self, player=0):
        """Play No Thanks against a CFR trained opponent"""
        history = ''
        self.player = player
        while True:
            [node_type, node] = self.game_tree.get_node(history)
            if node_type == 'chance':
                action = node.draw()
                if action is None:  # terminal checking is done in draw
                    break
            else:
                if self.player == node.state['current_player']:
                    print(node)
                    print(node.get_average_strategy())
                    # no error checking on input!
                    action = input('Pick an action - ' +
                                   ' or '.join(node.actions) + ': ')
                    print('\n\n\n\n')
                else:
                    strat = node.get_average_strategy()
                    # pick an action according to average strategy profile
                    action = choice(list(strat.keys()), p=list(strat.values()))
            history = history + ',' + str(action)
        print(node)
        print(self.game_tree.get_node(history)[1].scores)
        return history

    def cfr(self, history, i, k, probs):
        """Counter factual regret minimization algorithm"""
        opponent = (i+1) % self.game_tree.game_parameters['NUM_PLAYERS']
        [node_type, node] = self.game_tree.get_node(history)
        if node_type == 'chance':  
            a = node.draw()
            if a is None:  # terminal checking is done in draw
                return node.scores[i]-node.scores[opponent]
            new_history = history + ',' + str(a)
            return self.cfr(new_history, i, k, probs)
        else:
            node_utility = 0
            actions = node.actions
            action_utility = {a: 0 for a in actions}
            for a in actions:
                new_history = history + ',' + a
                new_probs = list(probs)  # copy list
                new_probs[node.state['current_player']] *= node.strat[a]
                action_utility[a] = self.cfr(new_history, i, k, new_probs)
                node_utility += node.strat[a]*action_utility[a]
            if node.state['current_player'] == i:
                for a in actions:
                    reg = action_utility[a] - node_utility
                    node.regret_table[a] += probs[opponent]*reg
                    node.strategy_table[a] += probs[i]*node.strat[a]
                node.set_strategy()
        return node_utility


class GameTree(dict):
    """Dictionary class that contains all nodes in the game"""
    game_parameters = {'NUM_PLAYERS': 2, 'STARTING_CHIPS': 2,
                       'CARD_NUMBERS': set(range(3, 8)), 'DECK_SIZE': 4}

    def __init__(self, game_parameters=None):
        super().__init__()
        if game_parameters is None:
            game_parameters = GameTree.game_parameters
        self.game_parameters = game_parameters

    def get_node(self, history):
        """Return game node given history key or make not if nonexistent"""
        # get a DrawNode if history is empty or last action 't' not terminal
        if not history or history[-1] == 't':
            if history not in self:
                self[history] = DrawNode(history, self)
            self[history].node_visit += 1
            return ['chance', self[history]]
        else:   # a player node
            if history not in self:
                self[history] = PlayerNode(history, self)
            self[history].node_visit += 1
            return ['player', self[history]]

    def get_score(self, history):
        """Get the game score from a player node with history"""
        # make last node to have final game state
        hist_str = history
        if hist_str not in self:
            self[hist_str] = PlayerNode(history, self)
        return self[hist_str].scores

    def save(self, filename=None):
        """Save game tree as pickle"""
        if filename is None:
            filename = str(self.game_parameters['STARTING_CHIPS']) + \
                '_' + ''.join(str(c) for c in
                              self.game_parameters['CARD_NUMBERS']) + \
                '_' + str(self.game_parameters['DECK_SIZE'])
        pickle.dump(self, open(filename + '.nothanks', 'wb'))

    def split_history_into_node_action(self, history):
        """Split history into last node and action taken by that node"""
        prev = history.rfind(',')
        last_action = history[prev+1:]
        return [self[history[:prev]], last_action]


class Node(object):
    """General game node for NoThanks"""
    def __init__(self, history=None, game_tree=None):
        self.history = history
        self.game_parameters = game_tree.game_parameters
        self.update_game_state(game_tree)
        self.calculate_score()
        self.node_visit = 0

    def update_game_state(self, game_tree):
        pass

    def calculate_score(self):
        """Calculate current score for this node for all players"""
        self.scores = [self.state[i]['chips']-sum([c for c in
                       self.state[i]['cards'] if c-1 not in
                       self.state[i]['cards']])
                       for i in range(self.game_parameters['NUM_PLAYERS'])]

    def __str__(self):
        """Return string version of this node. (string of state)"""
        string = ''
        for i in range(self.game_parameters['NUM_PLAYERS']):
            string = string + 'Player ' + str(i) + ' chips: ' + \
                                 str(self.state[i]['chips']) + '\n cards: '
            string = string + ' '.join(str(c) for c in self.state[i]['cards'])
            string = string + '\n'
        string = string + str(self.state['chips']) + ' chips on faceup ' + \
                          str(self.state['face_up'])
        return string


class DrawNode(Node):
    """Chance node that draws card from the deck"""

    def update_game_state(self, game_tree):
        """Update state of game from current history. Use previous nodes"""
        if self.history == '':
            self.state = {i: {'chips':
                              game_tree.game_parameters['STARTING_CHIPS'],
                              'cards': []}
                    for i in range(game_tree.game_parameters['NUM_PLAYERS'])}
            self.state['current_player'] = 0
            self.state['chips'] = 0  # chips on current card
            self.state['face_up'] = 0
            self.drawn_cards = set()
            self.is_terminal = False
        else:  # last action was 't'
            prev_node, last_action = \
                game_tree.split_history_into_node_action(self.history)
            self.state = copy.deepcopy(prev_node.state)
            self.drawn_cards = copy.deepcopy(prev_node.drawn_cards)
            curr_player = self.state['current_player']  # keep current player
            self.state[curr_player]['chips'] += self.state['chips']
            self.state[curr_player]['cards'].append(self.state['face_up'])
            self.state['chips'] = 0  # reset chips on card
            self.is_terminal = len(self.drawn_cards) == \
                               game_tree.game_parameters['DECK_SIZE']

    def draw(self):
        """Draw a card from remaining deck."""
        if not self.is_terminal:
            return random.sample(self.game_parameters['CARD_NUMBERS'] -
                                 self.drawn_cards, 1)[0]
        self.state['face_up'] = -1
        return None


class PlayerNode(Node):
    """Node that contain player actions and strategies"""

    def __init__(self, history=None, game_tree=None):
        """Initialize node with game history leading to this node"""
        super().__init__(history, game_tree)
        self.get_actions()
        self.initial_strategy()
        self.regret_table = {a: 0 for a in self.strat}
        self.strategy_table = {a: 0 for a in self.strat}

    def update_game_state(self, game_tree):
        """Update state of game from current history. Use previous nodes"""
        prev_node, previous_action = \
            game_tree.split_history_into_node_action(self.history)
        self.state = copy.deepcopy(prev_node.state)
        self.drawn_cards = copy.deepcopy(prev_node.drawn_cards)
        if previous_action == 'b':
            self.state['chips'] += 1  # add a chip to the card
            curr_player = self.state['current_player']
            self.state[curr_player]['chips'] -= 1
            self.state['current_player'] = (curr_player + 1) % \
                game_tree.game_parameters['NUM_PLAYERS']
        else:  # drawn card
            previous_action = int(previous_action)
            self.state['face_up'] = previous_action
            self.drawn_cards.add(previous_action)

    def get_actions(self):
        """Return available actions from this node"""
        cp = self.state['current_player']
        self.actions = ['t']
        # do not allow placing chip when chips = card value
        if self.state[cp]['chips'] > 0 and \
                self.state['face_up'] > self.state['chips']:
            self.actions.append('b')

    def initial_strategy(self):
        """Set initial probabilities of choosing each action"""
        self.strat = {a: 1/len(self.actions) for a in self.actions}

    def set_strategy(self):
        """Set strategy for this node using regret matching"""
        for a in self.regret_table:
            self.strat[a] = max(self.regret_table[a], 0)
        normalization = sum(self.strat.values())
        if normalization > 0:
            for a in self.regret_table:
                self.strat[a] /= normalization
        else:
            self.initial_strategy()

    def get_average_strategy(self):
        """Get average strategy over all CFR iterations"""
        normalization = sum(self.strategy_table.values())
        average_strategy = {a: 1/len(self.actions) for a in self.actions}
        if normalization > 0:
            for a in self.strategy_table:
                average_strategy[a] = self.strategy_table[a]/normalization
        return average_strategy

def main():
    trainer = NoThanks()
    start_time = time.time()
    trainer.train(iterations=1000)
    print(time.time() - start_time)

if __name__ == "__main__":
    main()
