# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:24:05 2017

@author: ACKWinDesk
"""
import random
import time
import copy
# several hacks, didn't realize dict could not use lits as keys so I just used 
# str(list) as the key instead

# make history and object and give it a .__str__ method

# Card points are negative, chip points are positive

class CfrTrainer(object):
    """Runs counter factual regret minimization training algorithm"""
    NUM_PLAYERS = 2

    def __init__(self):
        self.game_tree = GameTree()

    def train(self, iterations=10):
        """Runs cfr iterations"""
        for k in range(iterations):
            for i in range(self.NUM_PLAYERS):
                self.cfr([], i, k, self.NUM_PLAYERS*[1])

    def cfr(self, history, i, k, probs):  # p0, p1):
        """Counter factual regret minimization algorithm"""
        # check if history is terminal
        if DrawNode.check_if_terminal(history):
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
                    #action_utility[a] = self.cfr(new_history, i, k,
                    #                             p0, node.strat[a]*p1)
                node_utility += node.strat[a]*action_utility[a]
            if node.state['current_player'] == i:
                for a in actions:
                    reg = action_utility[a] - node_utility
                    node.regret_table[a] += probs[(i+1) % self.NUM_PLAYERS]*reg
                    node.strategy_table[a] += probs[i]*node.strat[a]
                node.set_strategy()
        return node_utility


class GameTree(object):
    def __init__(self):
        self.node_dict = dict()

    def get_node(self, history):
        """Return game node given history key or make not if nonexistent"""
        if not history or history[-1] == 't':  # empty history or card taken
            if str(history) not in self.node_dict:
                self.node_dict[str(history)] = DrawNode(history)
            new_history = history + [self.node_dict[str(history)].draw()]
            return ['chance', new_history]
        else:   # a player node
            if str(history) not in self.node_dict:
                self.node_dict[str(history)] = PlayerNode(history, self.node_dict)
            return ['player', self.node_dict[str(history)]]

    def get_score(self, history):
        """Get the game score from a player node with history"""
        # make last node to have final game state
        # not using get_node because it misinterprets 't' as game continuing
        if str(history) not in self.node_dict:
            self.node_dict[str(history)] = PlayerNode(history, self.node_dict)
        return self.node_dict[str(history)].calculate_score()


class DrawNode(object):
    CARD_NUMBERS = list(range(10))
    CARD_NUMBERS = set(CARD_NUMBERS[3:])
    # deck size is not length of cards because we don't play all
    DECK_SIZE = len(CARD_NUMBERS) - 3

    def __init__(self, history=None):
        self.history = history

    def draw(self):
        """Draw a card from remaining deck. Assumes this is allowed because
        we already checked if terminal"""
        drawn_cards = self.find_card_history(self.history)
        card = random.sample(DrawNode.CARD_NUMBERS - drawn_cards, 1)[0]
        return card

    @classmethod
    def check_if_terminal(cls, history):
        """Check if a history is terminal
        (deck has been drawn and last card taken)"""
        drawn_cards = cls.find_card_history(history)
        return len(drawn_cards) == DrawNode.DECK_SIZE and history[-1] == 't'

    @staticmethod
    def find_card_history(history):
        """Search history for drawn cards"""
        drawn_cards = set({card for card in history if isinstance(card, int)})
        return drawn_cards


class PlayerNode(object):
    NUM_PLAYERS = 2
    STARTING_CHIPS = 3

    def __init__(self, history=None, node_dict=None):
        self.history = history
        self.build_game_state(node_dict)
        # set strategy profile to even
        self.initial_strategy()
        # initialize regret and strategy tables
        self.regret_table = {a:0 for a in self.strat}
        self.strategy_table = {a:0 for a in self.strat}

    def build_game_state(self, node_dict):
        # this is probably redundant (and memory intensive) because I can parse
        # game history for this information but I'm lazy
        if not self.history[:-1]:  # if empty history (or just one drawn card)
            # initialize game state
            self.state = {i: {'chips':self.STARTING_CHIPS, 'cards': []} 
            for i in range(self.NUM_PLAYERS)}
            self.state['current_player'] = 0
            self.state['chips'] = 0  # chips on current card
            self.state['face_up'] = self.history[-1]
        else:
            previous_action = self.history[-1]
            if previous_action == 'b':  # last guy placed bet
                self.state =  copy.deepcopy(node_dict[
                                            str(self.history[:-1])].state)
                self.state['chips'] += 1  # add a chip to the card
                curr_player = self.state['current_player']
                self.state[curr_player]['chips'] -= 1
                self.state['current_player'] = (curr_player + 1) % \
                                               self.NUM_PLAYERS
            elif previous_action == 't':  # last action of game was take card
                two_actions_ago = self.history[-2]
                if two_actions_ago == 'b':  # bet on last card
                    self.state = copy.deepcopy(node_dict[
                                               str(self.history[:-1])].state)
                    self.state['chips'] += 1  # add a chip to the card
                    curr_player = self.state['current_player']
                    self.state[curr_player]['chips'] -= 1
                    self.state['current_player'] = (curr_player + 1) % \
                                                   self.NUM_PLAYERS
                else:  # card was drawn two actions ago
                    self.state = copy.deepcopy(node_dict[
                                               str(self.history[:-1])].state)
                    curr_player = self.state['current_player']
                self.state[curr_player]['cards'].append(self.state['face_up'])
                self.state[curr_player]['chips'] += self.state['chips']
                self.state['chips'] = 0  # reset chips on card
                self.state['face_up'] = -1  # signal this a terminal node 
            #elif isinstance(previous_action , int) 
            else:  # last action was drawing a new card, same current player
                self.state = copy.deepcopy(node_dict[
                                           str(self.history[:-2])].state)
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
        """Set probabilities of choosing each action"""
        actions = self.get_actions()
        self.strat = {a: 1/len(actions) for a in actions}

    def set_strategy(self):
        for a in self.regret_table:
            self.strat[a] = max(self.regret_table[a], 0)
        normalization = sum(self.strat.values())
        if normalization > 0:
            for a in self.regret_table:
                self.strat[a] /= normalization
        else:
            self.initial_strategy()

    def get_average_strategy(self):
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
                    for i in range(self.NUM_PLAYERS)]
        return scores

