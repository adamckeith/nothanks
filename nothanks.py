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
import re

# I'm worried that algorithm is not exactly zero sum so is missing that it might 
# be good for me for opponent to take card instead of adding to a sequence
# so, currently regret is only trying to get the best score, not beat opponent

# make history and object and give it a .__str__ method

# use information set because multiple histories lead to same game state

# GameTree has class variable game_parameters
# Card points are negative, chip points are positive
class CfrTrainer(object):
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

    def cfr(self, history, i, k, probs):
        """Counter factual regret minimization algorithm"""
        opponent = (i+1) % self.game_tree.game_parameters['NUM_PLAYERS']
        # check if history is terminal
        #if self.game_tree.check_if_terminal(history):
        #    score = self.game_tree.get_score(history)
            # zero-sum score is difference between player score and opponent
        #    return score[i]-score[opponent]
        [node_type, node] = self.game_tree.get_node(history)
        if node_type == 'chance':  
            a = node.draw()
            if a is None:  # terminal checking is done in draw
                return node.scores[i]-node.scores[opponent]
            #new_history = history + [a]
            new_history = history + ',' + str(a)
            return self.cfr(new_history, i, k, probs)
        else:
            node_utility = 0
            actions = node.actions # simplify this by list comp of strat
            action_utility = {a: 0 for a in actions}
            for a in actions:
                #new_history = history + [a]
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
#    game_parameters = {'NUM_PLAYERS': 2, 'STARTING_CHIPS': 2,
#                       'CARD_NUMBERS': set(range(3, 8)), 'DECK_SIZE': 4}
    game_parameters = {'NUM_PLAYERS': 2, 'STARTING_CHIPS': 2,
                       'CARD_NUMBERS': set(str(i) for i in range(3,8)), 
                       'DECK_SIZE': 4}

    def __init__(self, game_parameters=None):
        super().__init__()
        if game_parameters is None:
            game_parameters = GameTree.game_parameters
        self.game_parameters = game_parameters

    def get_node(self, history):
        """Return game node given history key or make not if nonexistent"""
        #hist_str = self.to_str(history)  # get string version of history
        hist_str = history
        # get a DrawNode if history is empty or last action 't' not terminal
        if not history or history[-1] == 't': #and not \
                #self.check_if_terminal(history):
            if hist_str not in self:
                self[hist_str] = DrawNode(history, self)
            return ['chance', self[hist_str]]
        else:   # a player node
            if hist_str not in self:
                self[hist_str] = PlayerNode(history, self)
            self[hist_str].node_visit += 1
            return ['player', self[hist_str]]

    def get_score(self, history):
        """Get the game score from a player node with history"""
        # make last node to have final game state
#        hist_str = self.to_str(history)  # get string version of history
        hist_str = history
        if hist_str not in self:
            self[hist_str] = PlayerNode(history, self)
        return self[hist_str].scores
        #return self[hist_str].calculate_score()

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

    def get_two_nodes_ago(self, history):
        prev_node, last_action = self.get_previous_node(history)
        return self.get_previous_node(prev_node.history)

    def check_if_terminal(self, history):
        """Check if a history is terminal
        (deck has been drawn and last card taken)"""
        self.game_tree.get_node(history)
        
        # Make this more efficient by first checking if last action is 't'
        # but now if condition is redundant when calling get_node
        if (not history or history[-1] == 't'):
            drawn_cards = DrawNode.find_card_history(history)
            return len(drawn_cards) == self.game_parameters['DECK_SIZE']
        return False

    @staticmethod
    def to_str(history):
        return ''.join(str(c) for c in history)


class Node(object):
    """General game node for NoThanks"""
    
    def __init__(self, history=None, game_tree=None):
        self.history = history
        self.game_parameters = game_tree.game_parameters
        self.update_game_state(game_tree)

    def update_game_state(self, game_tree):
        pass
    
    def calculate_score(self):
        """Calculate current score for this node for all players"""
        self.scores = [self.state[i]['chips']-sum([c for c in
                       self.state[i]['cards'] if c-1 not in
                       self.state[i]['cards']])
                       for i in range(self.game_parameters['NUM_PLAYERS'])]
    
class DrawNode(object):
    """Chance node that draws card from the deck"""

    def __init__(self, history=None, game_tree=None):
        """Create a DrawNode node"""
        # DrawNode does not need full history
        self.history = history
        self.game_parameters = game_tree.game_parameters
        self.update_game_state(game_tree)    
        self.calculate_score()
        # maybe make this like player node and call previous DrawNode 
        # for card history so don't search string every time
        #self.drawn_cards = self.find_drawn_cards(history)
        #self.drawn_cards = self.find_card_history(history)

    def update_game_state(self, game_tree):
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
            curr_player = self.state['current_player'] # keep current player
            self.state[curr_player]['chips'] += self.state['chips']
            self.state[curr_player]['cards'].append(int(self.state['face_up']))
            self.state['chips'] = 0  # reset chips on card
            self.is_terminal = len(self.drawn_cards) == \
                               game_tree.game_parameters['DECK_SIZE']           
        # return True if game can continue, False is terminal


                       
    def find_drawn_cards(self, history, game_tree):
        """Find drawn cards"""
        if not history:
            self.drawn_cards = set()
        else:
            self.drawn_cards = copy.deepcopy(game_tree.get_previous_node(
                                             history).drawn_cards)

    def draw(self):
        """Draw a card from remaining deck."""
        if not self.is_terminal:
            return random.sample(self.game_parameters['CARD_NUMBERS'] -
                             self.drawn_cards, 1)[0]
        return None
            
            

    @staticmethod
    def find_card_history(history):
        """Search history for drawn cards"""
        #drawn_cards = set({card for card in history if isinstance(card, int)})
        #drawn_cards = set(int(d) for d in re.findall(r'\d+', history))
        drawn_cards = set(re.findall(r'\d+', history))
        # convert CARD_NUMBERS to string do avoid this comprehension 
        # do integer conversion in draw? (maybe dont use ints at all)
        return drawn_cards


class PlayerNode(object):
    """Node that contain player actions and strategies"""

    def __init__(self, history=None, game_tree=None):
        """Initialize node with game history leading to this node"""
        self.history = history
        self.game_parameters = game_tree.game_parameters
        #self.is_terminal = False
        self.update_game_state(game_tree)
        #self.build_game_state(game_tree)
        #self.calculate_score()
        self.get_actions()
        # set strategy profile to even
        self.initial_strategy()
        # initialize regret and strategy tables
        self.regret_table = {a: 0 for a in self.strat}
        self.strategy_table = {a: 0 for a in self.strat}
        self.node_visit = 0

    def update_game_state(self, game_tree):
        prev_node, previous_action = \
            game_tree.split_history_into_node_action(self.history)
        self.state = copy.deepcopy(prev_node.state) 
        self.drawn_cards = copy.deepcopy(prev_node.drawn_cards)
        # last action was 'b' or draw()
        if previous_action == 'b': 
            self.state['chips'] += 1  # add a chip to the card
            curr_player = self.state['current_player']
            self.state[curr_player]['chips'] -= 1
            self.state['current_player'] = (curr_player + 1) % \
                game_tree.game_parameters['NUM_PLAYERS']
        else: # drawn card
            self.state['face_up'] = previous_action         
            self.drawn_cards.add(previous_action)
           
        
    def build_game_state(self, game_tree):
        """Build game state by adjusting state of previous PlayerNode in
        the history"""
        #history = self.convert_histstr_to_histlist(self.history)
        # this is probably redundant (and memory intensive) because I can parse
        # game history for this information but I'm lazy
        last_comma = self.history.rfind(',')
        if not last_comma:  # if only one card drawn
        #if not self.history[:-1]:  # if empty history (or just one drawn card)
            # initialize game state
            self.state = {i: {'chips':
                              self.game_parameters['STARTING_CHIPS'],
                              'cards': []}
                    for i in range(self.game_parameters['NUM_PLAYERS'])}
            self.state['current_player'] = 0
            self.state['chips'] = 0  # chips on current card
            #self.state['face_up'] = int(self.history[last_comma+1:])
            self.state['face_up'] = self.history[last_comma+1:]
            self.drawn_cards = set(int(self.state['face_up']))
            #self.state['face_up'] = self.history[-1]  # not right for string            
        else:  # there is a history
            penultimate_comma = self.history[:last_comma].rfind(',')
            previous_action = self.history[last_comma+1:]
            #previous_action = self.history[-1]
            if previous_action == 'b':  # last guy placed bet
                #[node_type, node] = game_tree.get_node(self.history[:-1])
                [node_type, node] = game_tree.get_node(self.history[:last_comma])
                self.state = copy.deepcopy(node.state)
                self.state['chips'] += 1  # add a chip to the card
                curr_player = self.state['current_player']
                self.state[curr_player]['chips'] -= 1
                self.state['current_player'] = (curr_player + 1) % \
                    self.game_parameters['NUM_PLAYERS']
            elif previous_action == 't':  # last action of game was take card
                # because only got here if terminal
                two_actions_ago = self.history[penultimate_comma+1:last_comma]
                #two_actions_ago = self.history[-2]
                [node_type, node] = game_tree.get_node(self.history[:last_comma])
                self.state = copy.deepcopy(node.state)
                self.drawn_cards = copy.deepcopy(node.drawn_cards)
                if two_actions_ago == 'b':  # bet on last card
                    #[node_type, node] = game_tree.get_node(self.history[:-1])
                    self.state['chips'] += 1  # add a chip to the card
                    curr_player = self.state['current_player']
                    self.state[curr_player]['chips'] -= 1
                    self.state['current_player'] = (curr_player + 1) % \
                        self.game_parameters['NUM_PLAYERS']
                else:  # card was drawn two actions ago
                    #[node_type, node] = game_tree.get_node(self.history[:-1])
                    curr_player = self.state['current_player']
                #self.state[curr_player]['cards'].append(self.state['face_up'])
                self.state[curr_player]['cards'].append(int(self.state['face_up']))
                self.state[curr_player]['chips'] += self.state['chips']
                self.state['chips'] = 0  # reset chips on card
                #self.state['face_up'] = -1  # signal this a terminal node
                self.state['face_up'] = '-1'  # signal this a terminal node
            else:  # last action was drawing a new card, 
                # get state from two actions ago (same current player)
                [node_type, node] = game_tree.get_node(self.history[:penultimate_comma])
                #[node_type, node] = game_tree.get_node(self.history[:-2])
                self.state = copy.deepcopy(node.state)
                curr_player = self.state['current_player']
                self.state[curr_player]['chips'] += self.state['chips']
                #self.state[curr_player]['cards'].append(self.state['face_up'])
                self.state[curr_player]['cards'].append(int(self.state['face_up']))
                self.state['chips'] = 0  # reset chips on card
                #self.state['face_up'] = int(self.history[last_comma+1:])
                self.state['face_up'] = self.history[last_comma+1:]
                #self.state['face_up'] = self.history[-1]

    def get_actions(self):
        """Return available actions from this node"""
        cp = self.state['current_player']
        self.actions = ['t']
        # do not allow placing chip when chips = card value
        if self.state[cp]['chips'] > 0 and \
                int(self.state['face_up']) > self.state['chips']:
                #self.state['face_up'] > self.state['chips']:
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

    @staticmethod
    def convert_histstr_to_histlist(hist_str):
        """quick hack to convert history string to list"""
        history = []
        if len(hist_str) == 0: 
            return history
        tsplit = hist_str.split('t')
        for i,t in enumerate(tsplit):
            if i>0:
                history.append('t')
            if not (t == '' or t==['']):       
                bsplit = t.split('b')
                for b in bsplit:
                    if b == '':
                        history.append('b')
                    else:
                        history.append(int(b))
        return history
        
class NoThanks(object):
    """Game No Thanks vs. trained AI"""
    def __init__(self, game_tree=None):
        self.game_tree = game_tree

    def play(self, player=0):
        """Play No Thanks against a CFR trained opponent"""
        self.history = ''
        self.player = player
        while True: #not self.game_tree.check_if_terminal(self.history):
            [node_type, node] = self.game_tree.get_node(self.history)
            if node_type == 'chance':  # here node is actually the new history
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
                #self.history = self.history + [action]
            self.history = self.history + ',' + str(action)
        print(self.game_tree.get_node(self.history)[1].scores)
        #score = self.game_tree.get_score(self.history)
        #print(score)
        #return score


def main():
    trainer = CfrTrainer()
    start_time = time.time()
    trainer.train(iterations=1000)
    print(time.time() - start_time)

if __name__ == "__main__":
    main()
