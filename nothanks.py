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
# several hacks, didn't realize dict could not use lits as keys so I just used 
# str(list) as the key instead

# make history and object and give it a .__str__ method

# use information set because multiple histories lead to same game state

# GameTree has class variable game_parameters
# Card points are negative, chip points are positive



def tostr(history):
    return ''.join(str(c) for c in history)

class CfrTrainer(object):
    """Runs counter factual regret minimization training algorithm"""
    NUM_PLAYERS = 2

    def __init__(self, game_parameters=None):
        self.game_tree = GameTree(game_parameters)

    def train(self, iterations=10):
        """Runs cfr iterations"""
        self.utils = [iterations*[0], iterations*[0]]
        for k in range(iterations):
            for i in range(self.NUM_PLAYERS):
                #self.cfr(History([]), i, k, self.NUM_PLAYERS*[1])
                self.utils[i][k] = self.cfr([], i, k, self.NUM_PLAYERS*[1])

    def cfr(self, history, i, k, probs):  # p0, p1):
        """Counter factual regret minimization algorithm"""
        # check if history is terminal
        if GameTree.check_if_terminal(history):
            return self.game_tree.get_score(history)[i]
        [node_type, node] = self.game_tree.get_node(history)
        if node_type == 'chance':  # here node is actually the new history
            return self.cfr(node, i, k, probs) #p0, p1)
        else:
            # initialize variables
            node_utility = 0
            actions = node.get_actions()  # simplify this by list comp of strat
            action_utility = {a: 0 for a in actions}
            for a in actions:
                new_history = history + [a]
                new_probs = list(probs)  # copy list
                new_probs[node.state['current_player']] *= node.strat[a]
                action_utility[a] = self.cfr(new_history, i, k, new_probs)
                node_utility += node.strat[a]*action_utility[a]
            if node.state['current_player'] == i:
                for a in actions:
                    reg = action_utility[a] - node_utility
                    node.regret_table[a] += probs[(i+1) %
                                   GameTree.game_parameters['NUM_PLAYERS']]*reg
                    node.strategy_table[a] += probs[i]*node.strat[a]
                node.set_strategy()
        return node_utility


class GameTree(dict):
    """Dictionary class that contains all nodes in the game"""
    game_parameters = {'NUM_PLAYERS': 2, 'STARTING_CHIPS': 2,
                       'CARD_NUMBERS': set(range(3, 8)), 'DECK_SIZE': 4}

    def __init__(self, game_parameters=None):
        super().__init__()
        if game_parameters is not None:
            GameTree.game_parameters = game_parameters
        
        
    def get_node(self, history):
        """Return game node given history key or make not if nonexistent"""
        # this doesn't get terminal nodes because they end in 't'
        if not history or history[-1] == 't':  # empty history or card taken
            if tostr(history) not in self:
                self[tostr(history)] = DrawNode(history)
            new_history = history + [self[tostr(history)].draw()]
            return ['chance', new_history]
        else:   # a player node
            if tostr(history) not in self:
                self[tostr(history)] = PlayerNode(history, self)
            self[tostr(history)].node_visit += 1
            return ['player', self[tostr(history)]]

    def get_score(self, history):
        """Get the game score from a player node with history"""
        # make last node to have final game state
        # not using get_node because it misinterprets 't' as game continuing
        if tostr(history) not in self:
            self[tostr(history)] = PlayerNode(history, self)
        return self[tostr(history)].calculate_score()

    def save_tree(self):
        """Save game tree as pickle"""
        pickle.dump(self, open('tree.nothanks', 'wb'))

    @staticmethod
    def check_if_terminal(history):
        return DrawNode.check_if_terminal(history)

class DrawNode(object):
    """Chance node that draws card from the deck"""

    def __init__(self, history=None):
        self.history = history

    def draw(self):
        """Draw a card from remaining deck. Assumes this is allowed because
        we already checked if terminal"""
        drawn_cards = self.find_card_history(self.history)
        card = random.sample(GameTree.game_parameters['CARD_NUMBERS'] -
                             drawn_cards, 1)[0]
        return card

    @classmethod
    def check_if_terminal(cls, history):
        """Check if a history is terminal
        (deck has been drawn and last card taken)"""
        drawn_cards = cls.find_card_history(history)
        return len(drawn_cards) == GameTree.game_parameters['DECK_SIZE'] and \
            history[-1] == 't'

    @staticmethod
    def find_card_history(history):
        """Search history for drawn cards"""
        drawn_cards = set({card for card in history if isinstance(card, int)})
        return drawn_cards


class PlayerNode(object):
    """Node that contain player actions and strategies"""
    
    def __init__(self, history=None, game_tree=None):
        """Initialize node with game history leading to this node"""
        self.history = history
        self.build_game_state(game_tree)
        # set strategy profile to even
        self.initial_strategy()
        # initialize regret and strategy tables
        self.regret_table = {a: 0 for a in self.strat}
        self.strategy_table = {a: 0 for a in self.strat}
        self.node_visit = 1

    def build_game_state(self, game_tree):
        """Build game state by adjusting state of previous PlayerNode in 
        the history"""
        # this is probably redundant (and memory intensive) because I can parse
        # game history for this information but I'm lazy
        if not self.history[:-1]:  # if empty history (or just one drawn card)
            # initialize game state
            self.state = {i: {'chips':
                              GameTree.game_parameters['STARTING_CHIPS'],
                              'cards': []}
                    for i in range(GameTree.game_parameters['NUM_PLAYERS'])}
            self.state['current_player'] = 0
            self.state['chips'] = 0  # chips on current card
            self.state['face_up'] = self.history[-1]
        else:
            previous_action = self.history[-1]
            if previous_action == 'b':  # last guy placed bet
                self.state =  copy.deepcopy(game_tree[
                                            tostr(self.history[:-1])].state)
                self.state['chips'] += 1  # add a chip to the card
                curr_player = self.state['current_player']
                self.state[curr_player]['chips'] -= 1
                self.state['current_player'] = (curr_player + 1) % \
                    GameTree.game_parameters['NUM_PLAYERS']
            elif previous_action == 't':  # last action of game was take card
                two_actions_ago = self.history[-2]
                if two_actions_ago == 'b':  # bet on last card
                    self.state = copy.deepcopy(game_tree[
                                               tostr(self.history[:-1])].state)
                    self.state['chips'] += 1  # add a chip to the card
                    curr_player = self.state['current_player']
                    self.state[curr_player]['chips'] -= 1
                    self.state['current_player'] = (curr_player + 1) % \
                        GameTree.game_parameters['NUM_PLAYERS']
                else:  # card was drawn two actions ago
                    self.state = copy.deepcopy(game_tree[
                                               tostr(self.history[:-1])].state)
                    curr_player = self.state['current_player']
                self.state[curr_player]['cards'].append(self.state['face_up'])
                self.state[curr_player]['chips'] += self.state['chips']
                self.state['chips'] = 0  # reset chips on card
                self.state['face_up'] = -1  # signal this a terminal node 
            else:  # last action was drawing a new card, same current player
                self.state = copy.deepcopy(game_tree[
                                           tostr(self.history[:-2])].state)
                curr_player = self.state['current_player']
                self.state[curr_player]['chips'] += self.state['chips']
                self.state[curr_player]['cards'].append(self.state['face_up'])
                self.state['chips'] = 0  # reset chips on card
                self.state['face_up'] = self.history[-1]

    def get_actions(self):
        """Return available actions from this node"""
        cp = self.state['current_player']
        actions = ['t']
        # do not allow placing chip when chips = card value
        if self.state[cp]['chips'] > 0 and \
                self.state['face_up'] > self.state['chips']:
            actions.append('b')
        return actions

    def initial_strategy(self):
        """Set initial probabilities of choosing each action"""
        actions = self.get_actions()
        self.strat = {a: 1/len(actions) for a in actions}

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
        average_strategy = {a: 1/len(self.get_actions())
                            for a in self.get_actions()}
        if normalization > 0:
            for a in self.strategy_table:
                average_strategy[a] = self.strategy_table[a]/normalization
        return average_strategy
        
        
    def calculate_score(self):
        """Calculate current score for this node for all players"""
        scores = [self.state[i]['chips']-sum([c for c in self.state[i]['cards']
                    if c-1 not in self.state[i]['cards']])
                    for i in range(GameTree.game_parameters['NUM_PLAYERS'])]
        return scores
        
    def __str__(self):
        """Return string version of this node. (string of state)"""
        string = ''
        for i in range(GameTree.game_parameters['NUM_PLAYERS']):
            string = string + 'Player ' + str(i) + ' chips: ' + \
                                 str(self.state[i]['chips']) + '\n cards: '
            string = string + ' '.join(str(c) for c in self.state[i]['cards'])
            string = string + '\n'
        string = string + str(self.state['chips']) + ' chips on faceup ' + \
                          str(self.state['face_up']) #+ '\n'
        return string


class NoThanks(object):
    """Game No Thanks vs. trained AI"""
    def __init__(self, game_tree=None, player=0):
        self.game_tree = game_tree
        self.player = player

    def play(self):
        """Play No Thanks against a CFR trained opponent"""
        self.history = []
        while not GameTree.check_if_terminal(self.history):
            [node_type, node] = self.game_tree.get_node(self.history)
            if node_type == 'chance':  # here node is actually the new history
                self.history = node  # draw a new card
            else:
                if self.player == node.state['current_player']:
                    print(node)
                    # no error checking on input!
                    action = input('Pick an action ' + 
                                   ' '.join(node.get_actions()) + ': ')
                    print('\n\n\n\n')
                else:
                    strat = node.get_average_strategy()
                    # pick an action according to average strategy profile
                    action = choice(list(strat.keys()), p=list(strat.values()))
                self.history = self.history + [action]
        print(self.game_tree[tostr(self.history)])
        print(self.game_tree.get_score(self.history))


def main():
    trainer = CfrTrainer()
    start_time = time.time()
    trainer.train(iterations=1000)
    print(time.time() - start_time)    

if __name__ == "__main__":
    main()


class History(list):
    """List subclass to overide __str__ method for dictionary hashing"""
    def __str__(self):
        return ''.join(str(c) for c in self)
    
    def __add__(self, other):
        """Add another list to this one. Convert other to History object"""
        return History(super().__add__(other))
    # slicing doesn't return History object
