"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from isolation import Board

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass
def open_move_score(game, player):
    """The basic evaluation function described in lecture that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

    
def custom_score(game, player):
    """The basic evaluation function that outputs a score
    equal to the number of moves open for your computer player on the board.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

    
def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function is called from within a Player instance as
    `self.score()`. It is not called directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
        
    depth : ## TODO
    
    time_left : ## TODO

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
        
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. This is a variation of the `custom_score()` function.

    Note: this function is called from within a Player instance as
    `self.score()`. It is not called directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)

    
class RandomPlayer():
    """Player that chooses a move randomly."""

    def get_move(self, game, time_left):
        """Randomly select a move from the available legal moves.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            A randomly selected legal move; may return (-1, -1) if there are
            no available legal moves.
        """
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        return legal_moves[random.randint(0, len(legal_moves) - 1)]



        
class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : function, callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score_2, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : function, callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            
            best_move = self.minimax(game, depth=self.search_depth)
                
        except SearchTimeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move


        
        
    def minimax(self, game, depth, maximizing_player=True):
        """Implement depth-limited minimax search algorithm as described in the 
        implementation uses the modified version of MINIMAX-DECISION in the AIMA text:
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            Helper functions added to this class: 
            
            
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        
        score, move = self.minimax_maxvalue(game, depth, self.time_left, maximizing_player)
        return move
        
        '''
        moves = game.get_legal_moves()
        best_move = None
        best_score = float('-inf')
        
        for move in moves:
            # 'move' is a tuple of tuples
            print(len(moves))
            # gamestate = game.forecast_move(move, game.get_active_player())
            gamestate = game.forecast_move(move)
            print(self.time_left())
            score = self.minimax_maxvalue(gamestate, depth, self.time_left, maximizing_player)
            if score > best_score:
                best_move = move
                print('best_move', best_move)
                best_score = score
                print('best_score', best_score)
                
                
        return best_move
        '''
    
        
    def minimax_maxvalue(self, game, depth, time_left, maximizing_player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or not game.get_legal_moves():
            return self.score(game, self), (-1, -1)

        moves = game.get_legal_moves()
        best_score = float('-inf')
        best_move = None
        for move in moves:
            gamestate = game.forecast_move(move)
            score, _ = self.minimax_minvalue(gamestate, depth-1, self.time_left, False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move

        
    def minimax_minvalue(self, game, depth, time_left, maximizing_player=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or not game.get_legal_moves():
            return self.score(game, self), (-1, -1)

        moves = game.get_legal_moves()
        best_score = float('inf')
        best_move = None
        for move in moves:
            gamestate = game.forecast_move(move)
            score, _  = self.minimax_maxvalue(gamestate, depth-1, self.time_left, maximizing_player)
            if score < best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    
        
        
        
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """
        
    
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        
        best_move = (-1, -1)
            
        if not game.get_legal_moves():
            return best_move
            
         
        try:
            d = 1
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            while True:
                ab_move = self.alphabeta(game, depth=d)
                if ab_move == (-1, -1):
                    break
                best_move = ab_move
                d += 1
                
         
        except SearchTimeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    
        
    def alphabeta(self, game, depth, alpha=float('-inf'), beta=float('inf'), maximizing_player=True):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        

        moves = game.get_legal_moves()
        # print(game.get_legal_moves().values()[0])
        best_move = None
        best_score = float('-inf')
       
        for move in moves:
            # 'move' is a tuple of tuples
            gamestate = game.forecast_move(move)
            score = self.alphabeta_maxvalue(gamestate, depth, self.time_left, alpha, beta, maximizing_player)
            if score > best_score:
                best_move = move
                print('best_move', best_move)
                best_score = score
                print('best_score', best_score)
                
        return best_move

        
    def alphabeta_maxvalue(self, game, depth, time_left, alpha, beta, maximizing_player):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or not game.get_legal_moves():
            return self.score(game, self)


        legal_moves = game.get_legal_moves()    
        #best_score = float("-inf")
        for move in legal_moves:
            # Forecast_move switches the active player
            next_state = game.forecast_move(move)
            score = self.alphabeta_minvalue(next_state, depth-1, self.time_left, alpha, beta, False)
            #score = self.score(game.forecast_move(move), self)
            if score > beta:
                return score
            alpha = max(alpha, score)
            
        return score
        
        
    def alphabeta_minvalue(self, game, depth, time_left, alpha, beta, maximizing_player=True):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if depth == 0 or not game.get_legal_moves():
            return self.score(game, self)

        legal_moves = game.get_legal_moves()    
        for move in legal_moves:
            # Forecast_move switches the active player
            next_state = game.forecast_move(move)
            score = self.alphabeta_maxvalue(next_state, depth-1, self.time_left, alpha, beta, True)
            #score = self.score(game.forecast_move(move), self)
            if score > alpha:
                return score
            beta = min(beta, score)
            
        return score