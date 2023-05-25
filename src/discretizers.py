from abc import ABC, abstractmethod
from collections import deque
from enum import Enum, auto
from functools import reduce
import operator
from typing import Dict, Any, Tuple

from gym.wrappers.order_enforcing import OrderEnforcing

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from src.utils.overcooked_explainability.utils import Orientations, Actions, get_move, CardinalDirection


class Held(Enum):
    NOTHING = auto()
    ONION = auto()
    TOMATO = auto()
    DISH = auto()
    SOUP = auto()


class PotState(Enum):
    FINISHED = auto()
    COOKING = auto()
    PREPARING = auto()
    NOT_STARTED = auto()


class Action(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    STAY = auto()
    INTERACT = auto()


class Direction(Enum):
    NORTH = auto()
    NORTHEAST = auto()
    EAST = auto()
    SOUTHEAST = auto()
    SOUTH = auto()
    SOUTHWEST = auto()
    WEST = auto()
    NORTHWEST = auto()


class Discretizer(ABC):
    NONE_VALUE = '*'
    PREDICATES = None

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        self.env: OrderEnforcing = env
        self.gridWorld: OvercookedGridworld = self.env.unwrapped.mdp
        self.valid_pos = self.gridWorld.get_valid_player_positions()
        self.sources = self.get_basic_sources()
        self.none_value = '*'
        self.predicate_space = self.get_predicate_space()
        self.num_states = reduce(operator.mul,
                                 [len(predicate_value) for predicate_value in self.predicate_space.values()])

        #self.initial_state = '-'.join([l2[0] for index, l2 in enumerate(list(self.predicate_space.values()))])

    def get_basic_sources(self):
        """
        Gets the basic sources

        :returns: Dictionary with all the basic sources of the layout
        """

        sources = {
            'onion': self.gridWorld.get_onion_dispenser_locations(),
            'tomato': self.gridWorld.get_tomato_dispenser_locations(),
            'dish': self.gridWorld.get_dish_dispenser_locations(),
            'service': self.gridWorld.get_serving_locations(),
            'pot': self.gridWorld.get_pot_locations()
        }

        # Bar zones, where the players can leave things on
        sources = {obj: list_pos for obj, list_pos in sources.items()}
        for obj, list_pos in sources.items():
            sources[obj] = [(pos, self._get_possible_orientations(pos)) for pos in list_pos]

        return sources

    @abstractmethod
    def get_predicates(self, obs: OvercookedState):
        """
        Gets all the predicates from actual state.

        :returns: Discretized state representation
        :param obs: Current Observation
        """

        # Held Predicate
        pass


    def get_predicate_space(self):
        """
        Gets the predicate space
        """
        num_pots = len(self.get_basic_sources()['pot'])
        predicate_space = {}
        for p, values in self.PREDICATES.items():
            if p == 'pot_state' or p == 'pot_pos':
                for i in range(num_pots):
                    predicate_space[p + '_' + str(i)] = values
            else:
                predicate_space[p] = values
        return predicate_space

    def get_held_predicate(self, obs: OvercookedState, player_id):
        """
        Gets the predicate 'Held'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """

        if obs is None:
            return {'held': self.predicate_space['held'][0]}
        player: PlayerState = obs.players[player_id]
        if player.has_object():
            held = player.get_object().name
            if held == 'onion':
                held = self.predicate_space['held'][1]
            elif held == 'tomato':
                held = self.predicate_space['held'][2]
            elif held == 'dish':
                held = self.predicate_space['held'][3]
            elif held == 'soup':
                held = self.predicate_space['held'][4]
        else:
            held = self.predicate_space['held'][0]
        return {'held ' + str(player_id): held}

    def get_pot_state_predicate(self, obs: OvercookedState):
        """
        Gets the predicate 'pot_state'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """

        if obs is None:
            return {f'pot {pos}': self.PREDICATES['pot_state'][0] for pos, ori in self.sources['pot']}
        unowned_objects = obs.unowned_objects_by_type
        oven = {f'pot {pos}': self.PREDICATES['pot_state'][0] for pos, ori in self.sources['pot']}
        oven_positions = [p for p, o in self.sources['pot']]
        if 'soup' in unowned_objects:
            soups = unowned_objects['soup']
            for soup in soups:
                if soup.position in oven_positions:
                    num_onions = soup.state[1]
                    pot_time = soup.state[2]
                    # Fi
                    if num_onions == 3 and pot_time == 20:
                        oven[f'pot {soup.position}'] = self.PREDICATES['pot_state'][1]
                    # Co
                    elif num_onions == 3:
                        oven[f'pot {soup.position}'] = self.PREDICATES['pot_state'][2]
                    # Wa
                    else:
                        oven[f'pot {soup.position}'] = self.PREDICATES['pot_state'][3]

        return oven

    def get_onion_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest onion as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['onion'] = self.PREDICATES['onion_pos'][0]
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['onion'] = self.get_action_to_nearest_object(player.position, Orientations(player.orientation), 'onion',
                                                      temporary_sources)
            pos_predicate['onion'] = pos_predicate['onion'].name[0].upper()
            return pos_predicate

    def get_tomato_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest tomato as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['tomato'] = self.PREDICATES['tomato_pos'][0]
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['tomato'] = self.get_action_to_nearest_object(player.position,
                                                                       Orientations(player.orientation), 'tomato',
                                                                       temporary_sources)
            pos_predicate['tomato'] = pos_predicate['tomato'].name[0].upper()
            return pos_predicate

    def get_dish_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest dish as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['dish'] = self.PREDICATES['dish_pos'][0]
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['dish'] = self.get_action_to_nearest_object(player.position,
                                                                       Orientations(player.orientation), 'dish',
                                                                       temporary_sources)
            pos_predicate['dish'] = pos_predicate['dish'].name[0].upper()
            return pos_predicate

    def get_service_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest service as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['service'] = self.PREDICATES['service_pos'][0]
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['service'] = self.get_action_to_nearest_object(player.position,
                                                                       Orientations(player.orientation), 'service',
                                                                       temporary_sources)
            pos_predicate['service'] = pos_predicate['service'].name[0].upper()
            return pos_predicate

    def get_soup_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest soup as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['soup'] = self.PREDICATES['soup_pos'][0]
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['soup'] = self.get_action_to_nearest_soup(player.position,
                                                                       Orientations(player.orientation), 'soup',
                                                                       temporary_sources)
            pos_predicate['soup'] = pos_predicate['soup'].name[0].upper()
            return pos_predicate

    def get_pot_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest pot as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        for i in range(len(self.sources['pot'])):
            if obs is None:
                pos_predicate[f'pot {i}'] = self.PREDICATES['pot_pos'][0]
            else:
                player: PlayerState = obs.players[player_id]
                pos_predicate[f'pot {i}'] = self.get_action_to_nearest_pot(player.position,
                                                                        Orientations(player.orientation), 'pot',
                                                                        temporary_sources, i)
                pos_predicate[f'pot {i}'] = pos_predicate[f'pot {i}'].name[0].upper()
        return pos_predicate

    def get_partner_pos_predicate(self, obs: OvercookedState, player_id, partner_id):
        """ Computes next action to get the nearest onion as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['partner_pos'] = self.PREDICATES['partner_pos'][0]
            return pos_predicate
        else:
            pos_predicate['partner_pos'] = self.get_action_to_player(obs, player_id, partner_id)
            pos_predicate['partner_pos'] = pos_predicate['partner_pos'].name[0].upper()
            return pos_predicate

    def get_partner_zone_predicate(self, obs: OvercookedState, player_id, partner_id):
        """ Computes next action to get the nearest onion as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['partner_zone'] = self.PREDICATES['partner_zone'][0]
            return pos_predicate
        else:
            pos_predicate['partner_zone'] = self.get_partner_zone(obs, player_id, partner_id)
            pos_predicate['partner_zone'] = pos_predicate['partner_zone'].value
            return pos_predicate

    def get_temporary_sources(self, obs):
        if obs is not None:
            unowned_objects = obs.unowned_objects_by_type
            # Temporary sources (coordination)
            temporary_sources = {
                obj: [(obj_i.position, self._get_possible_orientations(obj_i.position)) for obj_i in list_obj] for
                obj, list_obj in unowned_objects.items()}
            # Union between self.sources and temporary_sources
            for obj, l in self.sources.items():
                if obj in temporary_sources:
                    temporary_sources[obj] += l
                else:
                    temporary_sources[obj] = l
            return temporary_sources
        else:
            return None

    @staticmethod
    def get_possible_actions():
        """
        Returns the number of possible actions that a player can take.
        """
        return Actions

    def get_predicate_options(self, predicate_example, indentation=0):
        """
        :returns: All the possible actions in a string
        :params indentation: Number of right identations to add in the text
        """
        num_pots = len(self.get_basic_sources()['pot'])
        text_indentation = '\t' * indentation
        if indentation == 0:
            text_indentation = ''
        text = ''
        for predicate, options in self.predicate_space.items():

            #if predicate == 'pot_state' or predicate == 'pot_pos':
            #    text = text + ''.join([text_indentation + '+ ' + predicate + f' {i}\t' + '|'.join(options) + '\n' for i in range(num_pots)])
            if predicate == 'held':
                text = text + text_indentation + '+ ' + predicate + '\t\t' + ' | '.join(options) + '\n'
            else:
                text = text + text_indentation + '+ ' + predicate + '\t' + ' | '.join(options) + '\n'

        text = text + '\n' + text_indentation + 'Example:\t' + predicate_example + '\n'
        return text

    def get_action_options(self, indentation=0):
        """
        :returns: All the possible actions in a string
        :params indentation: Number of right identations to add in the text
        """
        text_indentation = '\t' * indentation
        if indentation == 0:
            text_indentation = ''
        text = ''
        for action in Actions:
           text = text + text_indentation + '+ ' + action.name + '\n'

        # Give an example
        text = text + '\n' + text_indentation + 'Example:\t' + Actions(1).name + '\n'
        return text

    def get_predicate_label(self, predicate):
        """
        Converts the predicate to a label. Used for nicer visualization.
        """
        text_split = predicate.split('-')
        return '-'.join(text_split)

    def is_predicate(self, predicate):
        """
        :returns: True if predicate is valid
        :param predicate: Predicate
        """
        predicate_split = predicate.split('-')
        for valid_predicate, option in zip(list(self.predicate_space.keys()), predicate_split):
            # FIXME: Review if
            if option not in self.predicate_space[valid_predicate]:
                # print('Option', option, 'is not valid for predicate', valid_predicate)
                return False
        return True

    def is_action(self, action):
        """
        :returns: True if action is valid
        """
        if action in [a.name for a in Actions]:
            return True
        else:
            return False

    def get_num_possible_states(self):
        """
        :returns: the number of different states that exists.
        """
        return reduce(operator.mul, [len(option_list) for predicate, option_list in self.predicate_space.items()])

    @staticmethod
    def bfs(grid, start, goal, clear):
        """
        Shortest Path using BFS algorithm.

        :returns: Shortest path between position start to goal
        :param grid: Matrix layout
        :param start: Start position
        :param goal: Goal position
        :param clear: Types of cell where the agent can go through.
        """

        queue = deque([[start]])
        seen = set([start])
        width = len(grid[0])
        height = len(grid)
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if goal[0] == x and goal[1] == y:
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # Valid position to visit later
                if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] in clear and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    @staticmethod
    def get_moves(positions, end_ori: Orientations, agent_ori: Orientations):
        """
        Converts a list of positions in a list of actions.

        :returns: List of actions.
        :param positions: List of positions
        :param ori: Final orientation
        :param agent_ori: Initial agent orientation
        """

        if positions == None:
            return None
        actions = []
        # Convert each pair of position to an action
        for i in range(len(positions) - 1):
            actions.append(get_move(positions[i], positions[i + 1]))

        # If needed, add one step to face well the source
        #print('actions', actions, 'ori_source', ori.name, 'ori_agent', agent_ori)
        if len(actions) >= 1 and actions[-1].name != Orientations(end_ori).name or len(actions) == 0 and end_ori != agent_ori:
            actions.append(Actions[end_ori.name])

        # At the end, add the interaction action
        actions.append(Actions.Interact)
        return actions

    def get_action_to_nearest_object(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the object.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources.
        """

        # Assert that exist some source for our object
        assert obj in sources, f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        # Get all possible source positions
        obj_pos_list = [((p[0] - orient[0], p[1] - orient[1]), Orientations(orient)) for p, orient in sources[obj]]
        # Get path to each source position
        paths = [self.get_moves(self.bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori) for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Actions.Stay
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def get_action_to_player(self, obs, player_id, partner_id):
        """
        Gets the next action to perform in order to be closer to the object.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources.
        """

        player: PlayerState = obs.players[player_id]
        partner: PlayerState = obs.players[partner_id]

        clear = [" ", "X"]
        # Get all possible source positions
        #obj_pos_list = [((p[0] - orient[0], p[1] - orient[1]), Orientations(orient)) for p, orient in sources[obj]]
        # Get path to each source position
        path = self.get_moves(
                    self.bfs(self.gridWorld.terrain_mtx, start=player.position, goal=partner.position, clear=clear),
                    end_ori=Orientations(partner.orientation),
                    agent_ori=Orientations(player.orientation))

        # If the agent can't take nothing then stay
        if path is None:
            print('Shortest Path: Not existing path')
            return Actions.Stay

        # Remove interact
        path = path[:len(path)-1]
        # Take the shortest path and return the first action
        distance = len(path)
        print('Distance to partner:', distance, 'Shortest Path:', path)
        return path[0]

    def get_action_to_nearest_pot(self, pos, ori: Orientations, obj, sources, id):
        """
        Gets the next action to perform in order to be closer to the nearest pot.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        :param id: Pot id
        """

        # Assert that exist some source for our object
        assert obj in sources, f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        pot = sources[obj][id]
        # Get all possible source positions
        obj_pos_list = [((pot[0][0] - pot[1][0], pot[0][1] - pot[1][1]), Orientations(pot[1]))]
        # Get path to each source position
        paths = [self.get_moves(self.bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori) for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Actions.Nothing
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def get_action_to_nearest_soup(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the nearest soup.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        """

        # Assert that exist some source for our object
        if obj not in sources:
            return Actions.Stay
        clear = [" "]
        # Get all possible source positions
        pot_pos_list = [p for p, o in self.sources['pot']]
        obj_pos_list = [((p[0] - orient[0], p[1] - orient[1]), Orientations(orient)) for p, orient in sources[obj] if p not in pot_pos_list]

        # Get path to each source position
        paths = [self.get_moves(self.bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori) for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Actions.Stay
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def str_predicate_to_dict(self, predicate: str):
        predicate_space = self.get_predicate_space()
        if predicate is None:
            return None
        return dict(zip(list(predicate_space.keys()), predicate.split('-')))

    def dict_predicate_to_str(self, predicate: dict):
        return '-'.join(list(predicate.values()))

    def _get_possible_orientations(self, position):
        """
        Returns which is the correct orientation to interact well with position.

        :param position: Position with which we want to interact.
        """
        possible_orientations = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for ori in possible_orientations:
            next_pos = (position[0] + ori[0], position[1] + ori[1])
            if next_pos in self.valid_pos:
                return (-ori[0], -ori[1])

        raise Exception('Position {} is not accessible to interact'.format(position))

    def get_partner_zone(self, obs, player_id, partner_id):
        """ Computes in which zone is the partner located:

        :return: A CardinalDirection
        """
        player_pos = obs.players[player_id].position
        partner_pos = obs.players[partner_id].position

        if player_pos[0] == partner_pos[0] and player_pos[1] > partner_pos[1]:
            return CardinalDirection.North
        elif player_pos[0] == partner_pos[0] and player_pos[1] < partner_pos[1]:
            return CardinalDirection.South
        elif player_pos[0] > partner_pos[0] and player_pos[1] == partner_pos[1]:
            return CardinalDirection.West
        elif player_pos[0] < partner_pos[0] and player_pos[1] == partner_pos[1]:
            return CardinalDirection.East

        elif player_pos[0] > partner_pos[0] and player_pos[1] > partner_pos[1]:
            return CardinalDirection.NorthWest
        elif player_pos[0] < partner_pos[0] and player_pos[1] < partner_pos[1]:
            return CardinalDirection.SouthEast
        elif player_pos[0] > partner_pos[0] and player_pos[1] < partner_pos[1]:
            return CardinalDirection.SouthWest
        else:
            return CardinalDirection.NorthEast


class OvercookedDiscretizer:
    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        self.env: OrderEnforcing = env
        self.gridWorld: OvercookedGridworld = self.env.unwrapped.mdp
        self.valid_pos = self.gridWorld.get_valid_player_positions()
        self.sources = self._get_basic_sources()

    def _get_possible_orientations(self, position):
        """
        Returns which is the correct orientation to interact well with position.

        :param position: Position with which we want to interact.
        """
        possible_orientations = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for ori in possible_orientations:
            next_pos = (position[0] + ori[0], position[1] + ori[1])
            if next_pos in self.valid_pos:
                return (-ori[0], -ori[1])

        raise Exception('Position {} is not accessible to interact'.format(position))

    def _get_basic_sources(self):
        """
        Gets the basic sources

        :returns: Dictionary with all the basic sources of the layout
        """

        sources = {
            'onion': self.gridWorld.get_onion_dispenser_locations(),
            'tomato': self.gridWorld.get_tomato_dispenser_locations(),
            'dish': self.gridWorld.get_dish_dispenser_locations(),
            'service': self.gridWorld.get_serving_locations(),
            'pot': self.gridWorld.get_pot_locations()
        }

        # Bar zones, where the players can leave things on
        sources = {obj: list_pos for obj, list_pos in sources.items()}
        for obj, list_pos in sources.items():
            sources[obj] = [(pos, self._get_possible_orientations(pos)) for pos in list_pos]

        return sources

    def _get_temporary_sources(self, obs):
        if obs is not None:
            unowned_objects = obs.unowned_objects_by_type
            # Temporary sources (coordination)
            temporary_sources = {
                obj: [(obj_i.position, self._get_possible_orientations(obj_i.position)) for obj_i in list_obj] for
                obj, list_obj in unowned_objects.items()}
            # Union between self.sources and temporary_sources
            for obj, l in self.sources.items():
                if obj in temporary_sources:
                    temporary_sources[obj] += l
                else:
                    temporary_sources[obj] = l
            return temporary_sources
        else:
            return None

    @staticmethod
    def _bfs(grid, start, goal, clear):
        """
        Shortest Path using BFS algorithm.

        :returns: Shortest path between position start to goal
        :param grid: Matrix layout
        :param start: Start position
        :param goal: Goal position
        :param clear: Types of cell where the agent can go through.
        """

        queue = deque([[start]])
        seen = set([start])
        width = len(grid[0])
        height = len(grid)
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if goal[0] == x and goal[1] == y:
                return path
            for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # Valid position to visit later
                if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] in clear and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))

    @staticmethod
    def _get_move(start_pos, end_pos):
        if start_pos[0] < end_pos[0]:
            return Action.RIGHT
        if start_pos[0] > end_pos[0]:
            return Action.LEFT
        if start_pos[1] < end_pos[1]:
            return Action.BOTTOM
        if start_pos[1] > end_pos[1]:
            return Action.TOP
        else:
            return Action.STAY

    @staticmethod
    def _get_moves(positions, end_ori: Orientations, agent_ori: Orientations):
        """
        Converts a list of positions in a list of actions.

        :returns: List of actions.
        :param positions: List of positions
        :param ori: Final orientation
        :param agent_ori: Initial agent orientation
        """

        if positions == None:
            return None
        actions = []
        # Convert each pair of position to an action
        for i in range(len(positions) - 1):
            actions.append(OvercookedDiscretizer._get_move(positions[i], positions[i + 1]))

        # If needed, add one step to face well the source
        #print('actions', actions, 'ori_source', ori.name, 'ori_agent', agent_ori)
        if len(actions) >= 1 and actions[-1].name != Orientations(end_ori).name or len(actions) == 0 and end_ori != agent_ori:
            actions.append(Action[end_ori.name.upper()])

        # At the end, add the interaction action
        actions.append(Action.INTERACT)
        return actions

    @abstractmethod
    def _get_held_predicate(self, obs: OvercookedState, player_id):
        pass

    @abstractmethod
    def _get_pot_state_predicate(self, obs: OvercookedState):
        pass

    @abstractmethod
    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        pass

    @abstractmethod
    def dict_to_tuple(self, predicates: dict):
        pass

    @abstractmethod
    def tuple_to_dict(self, predicates: tuple):
        pass


class Discretizer10(OvercookedDiscretizer):
    """ Discretizer 10

    Predicates:
        - held
        - pot_state
    """

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)
        self.predicate_space: Dict[str, Enum]
        self.predicate_space = {'held 0':  Held}
        self.predicate_space.update({f'pot {pos[0]} state':  PotState for pos in self.sources['pot']})

    def _get_held_predicate(self, obs: OvercookedState, player_id):
        """
        Gets the predicate 'Held'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """

        if obs is None:
            return {'held': Held.NOTHING}
        player: PlayerState = obs.players[player_id]
        if player.has_object():
            held = player.get_object().name
            if held == 'onion':
                held = Held.ONION
            elif held == 'tomato':
                held = Held.TOMATO
            elif held == 'dish':
                held = Held.DISH
            elif held == 'soup':
                held = Held.SOUP
        else:
            held = Held.NOTHING
        return {'held ' + str(player_id): held}

    def _get_pot_state_predicate(self, obs: OvercookedState):
        """
        Gets the predicate 'pot_state'

        :returns: Dictionary with the predicate
        :param obs: Current Observation
        """

        if obs is None:
            return {f'pot {pos} state': PotState.NOT_STARTED for pos, ori in self.sources['pot']}
        unowned_objects = obs.unowned_objects_by_type
        oven = {f'pot {pos} state': PotState.NOT_STARTED for pos, ori in self.sources['pot']}
        oven_positions = [p for p, o in self.sources['pot']]
        if 'soup' in unowned_objects:
            soups = unowned_objects['soup']
            for soup in soups:
                if soup.position in oven_positions:
                    num_onions = soup.state[1]
                    pot_time = soup.state[2]
                    # Fi
                    if num_onions == 3 and pot_time == 20:
                        oven[f'pot {soup.position} state'] = PotState.FINISHED
                    # Co
                    elif num_onions == 3:
                        oven[f'pot {soup.position} state'] = PotState.COOKING
                    # Wa
                    else:
                        oven[f'pot {soup.position} state'] = PotState.PREPARING

        return oven

    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        obs = obs[1]

        # Held Predicate
        held = self._get_held_predicate(obs, 0)

        # Pot state Predicate
        pot_state = self._get_pot_state_predicate(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(held)
        predicate.update(pot_state)

        if to_dict:
            return predicate
        else:
            return tuple(predicate[k] for k in predicate)

    def dict_to_tuple(self, predicates: dict):
        if isinstance(predicates, tuple): return predicates
        preds = [predicates['held 0']]
        preds.extend(predicates[f'pot {pos[0]}'] for pos in self.sources['pot'])
        return tuple(preds)

    def tuple_to_dict(self, predicates: tuple):
        if isinstance(predicates, dict): return predicates
        preds = {'held 0': predicates[0]}
        preds.update({f'pot {pos[0]}': predicates[1+i] for i, pos in enumerate(self.sources['pot'])})
        return preds


class Discretizer11(Discretizer10):
    """Discretizer 11

    Predicates:
        - held
        - pot_state
        - predicate_pos
    """

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)
        self.predicate_space: Dict[str, Any]
        self.predicate_space.update({
            'onion':   Action,
            'tomato':  Action,
            'soup':    Action,
            'dish':    Action,
            'service': Action
        })
        self.predicate_space.update({f'pot {pos[0]} pos':  Action for pos in self.sources['pot']})

    def _get_action_to_nearest_object(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the object.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources.
        """

        # Assert that exist some source for our object
        assert obj in sources, f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        # Get all possible source positions
        obj_pos_list = [((p[0] - orient[0], p[1] - orient[1]), Orientations(orient))
                        for p, orient in sources[obj]]
        # Get path to each source position
        paths = [self._get_moves(self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori)
                 for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Action.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_action_to_nearest_pot(self, pos, ori: Orientations, obj, sources, id):
        """
        Gets the next action to perform in order to be closer to the nearest pot.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        :param id: Pot id
        """

        # Assert that exist some source for our object
        assert obj in sources, f"Error: Can't find object '{obj}' in sources. Existent objects are {list(sources.keys())}"
        clear = [" "]
        pot = sources[obj][id]
        # Get all possible source positions
        obj_pos_list = [((pot[0][0] - pot[1][0], pot[0][1] - pot[1][1]), Orientations(pot[1]))]
        # Get path to each source position
        paths = [self._get_moves(self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori)
                 for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Action.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_action_to_nearest_soup(self, pos, ori: Orientations, obj, sources):
        """
        Gets the next action to perform in order to be closer to the nearest soup.

        :returns: Next action
        :param pos: Actual agent position
        :param ori: Actual agent orientation
        :param obj: Object that the agent want to achieve
        :param sources: Current layout sources
        """

        # Assert that exist some source for our object
        if obj not in sources:
            return Action.STAY
        clear = [" "]
        # Get all possible source positions
        pot_pos_list = [p for p, o in self.sources['pot']]
        obj_pos_list = [((p[0] - orient[0], p[1] - orient[1]), Orientations(orient))
                        for p, orient in sources[obj] if p not in pot_pos_list]

        # Get path to each source position
        paths = [self._get_moves(self._bfs(self.gridWorld.terrain_mtx, pos, goal=obj_pos, clear=clear), orient, ori)
                 for obj_pos, orient in obj_pos_list]
        # Remove all None paths
        paths = [path for path in paths if path != None]
        # If the agent can't take nothing then stay
        if len(paths) == 0:
            #print(obj, 'Shortest Path: Not existing path')
            return Action.STAY
        # Take the shortest path and return the first action
        lengths = list(map(len, paths))
        min_index = lengths.index(min(lengths))
        shortest_path = paths[min_index]
        #print(obj, '--', pos, ori.name, 'Shortest Path:', shortest_path)
        return shortest_path[0]

    def _get_object_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id, object: str):
        """ Computes next action to get the nearest of the object as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate[object] = Action.STAY
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate[object] = self._get_action_to_nearest_object(player.position,
                                                                       Orientations(player.orientation),
                                                                       object,
                                                                       temporary_sources)
            return pos_predicate

    def _get_pot_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest pot as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        for i in range(len(self.sources['pot'])):
            if obs is None:
                pos_predicate[f'pot {self.sources["pot"][i][0]} pos'] = Action.STAY
            else:
                player: PlayerState = obs.players[player_id]
                pos_predicate[f'pot {self.sources["pot"][i][0]} pos'] = \
                    self._get_action_to_nearest_pot(player.position,
                                                    Orientations(player.orientation),
                                                    'pot',
                                                    temporary_sources,
                                                    i)
        return pos_predicate

    def _get_soup_pos_predicate(self, obs: OvercookedState, temporary_sources, player_id):
        """ Computes next action to get the nearest soup as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['soup'] = Action.STAY
            return pos_predicate
        else:
            player: PlayerState = obs.players[player_id]
            pos_predicate['soup'] = self._get_action_to_nearest_soup(player.position,
                                                                     Orientations(player.orientation),
                                                                     'soup',
                                                                     temporary_sources)
            return pos_predicate

    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        obs = obs[1]

        # Temporary sources (coordination)
        temporary_sources = self._get_temporary_sources(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(self._get_held_predicate(obs, 0))
        predicate.update(self._get_pot_state_predicate(obs))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='onion')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='tomato')
        )
        predicate.update(self._get_soup_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='dish')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='service')
        )
        predicate.update(self._get_pot_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))

        if to_dict:
            return predicate
        else:
            return tuple(predicate[k] for k in predicate)

    def dict_to_tuple(self, predicates: dict):
        if isinstance(predicates, tuple): return predicates
        preds = [predicates['held 0']]
        preds.extend(predicates[f'pot {pos[0]} state'] for pos in self.sources['pot'])
        preds.extend((predicates['onion'], predicates['tomato'], predicates['soup'],
                      predicates['dish'], predicates['service']))
        preds.extend(predicates[f'pot {pos[0]} pos'] for pos in self.sources['pot'])

        return tuple(preds)

    def tuple_to_dict(self, predicates: tuple):
        if isinstance(predicates, dict): return predicates
        preds = {'held 0': predicates[0]}
        preds.update({f'pot {pos[0]} state': predicates[1+i] for i, pos in enumerate(self.sources['pot'])})
        preds.update({
            'onion':   predicates[1+len(self.sources['pot'])+0],
            'tomato':  predicates[1+len(self.sources['pot'])+1],
            'soup':    predicates[1+len(self.sources['pot'])+2],
            'dish':    predicates[1+len(self.sources['pot'])+3],
            'service': predicates[1+len(self.sources['pot'])+4]
        })
        preds.update({f'pot {pos[0]} pos': predicates[1+len(self.sources['pot'])+5+i]
                      for i, pos in enumerate(self.sources['pot'])})
        return preds


class Discretizer12(Discretizer11):
    """Discretizer

    Predicates:
        - held
        - held_partner
        - pot_state
        - predicate_pos
    """

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)
        self.predicate_space.update({'held 1':  Held})

    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        obs = obs[1]

        # Temporary sources (coordination)
        temporary_sources = self._get_temporary_sources(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(self._get_held_predicate(obs, 0))
        predicate.update(self._get_held_predicate(obs, 1))
        predicate.update(self._get_pot_state_predicate(obs))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='onion')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='tomato')
        )
        predicate.update(self._get_soup_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='dish')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='service')
        )
        predicate.update(self._get_pot_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))

        if to_dict:
            return predicate
        else:
            return tuple(predicate[k] for k in predicate)

    def dict_to_tuple(self, predicates: dict):
        if isinstance(predicates, tuple): return predicates
        preds = [predicates['held 0'], predicates['held 1']]
        preds.extend(predicates[f'pot {pos[0]} state'] for pos in self.sources['pot'])
        preds.extend((predicates['onion'], predicates['tomato'], predicates['soup'],
                      predicates['dish'], predicates['service']))
        preds.extend(predicates[f'pot {pos[0]} pos'] for pos in self.sources['pot'])

        return tuple(preds)

    def tuple_to_dict(self, predicates: tuple):
        if isinstance(predicates, dict): return predicates
        preds = {'held 0': predicates[0], 'held 1': predicates[1]}
        preds.update({f'pot {pos[0]} state': predicates[2+i] for i, pos in enumerate(self.sources['pot'])})
        preds.update({
            'onion':   predicates[2+len(self.sources['pot'])+0],
            'tomato':  predicates[2+len(self.sources['pot'])+1],
            'soup':    predicates[2+len(self.sources['pot'])+2],
            'dish':    predicates[2+len(self.sources['pot'])+3],
            'service': predicates[2+len(self.sources['pot'])+4]
        })
        preds.update({f'pot {pos[0]} pos': predicates[2+len(self.sources['pot'])+5+i]
                      for i, pos in enumerate(self.sources['pot'])})
        return preds


class Discretizer13(Discretizer11):
    """Discretizer

    Predicates:
        - held
        - pot_state
        - predicate_pos
        - partner_zone
    """

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)
        self.predicate_space.update({'partner_zone':  Direction})

    def _get_partner_zone(self, obs, player_id, partner_id):
        """ Computes in which zone is the partner located:

        :return: A CardinalDirection
        """
        player_pos = obs.players[player_id].position
        partner_pos = obs.players[partner_id].position

        if player_pos[0] == partner_pos[0] and player_pos[1] > partner_pos[1]:
            return Direction.NORTH
        elif player_pos[0] == partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTH
        elif player_pos[0] > partner_pos[0] and player_pos[1] == partner_pos[1]:
            return Direction.WEST
        elif player_pos[0] < partner_pos[0] and player_pos[1] == partner_pos[1]:
            return Direction.EAST

        elif player_pos[0] > partner_pos[0] and player_pos[1] > partner_pos[1]:
            return Direction.NORTHWEST
        elif player_pos[0] < partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTHEAST
        elif player_pos[0] > partner_pos[0] and player_pos[1] < partner_pos[1]:
            return Direction.SOUTHWEST
        else:
            return Direction.NORTHEAST

    def _get_partner_zone_predicate(self, obs: OvercookedState, player_id, partner_id):
        """ Computes next action to get the nearest onion as fast as possible.

        :param obs: Actual observation of the environment.
        :param temporary_sources: original object sources + unowned objects
        :param player_id: Next action from position and orientation of player with id=player_id
        :return: Next action to get the object as fast as possible.
        """
        pos_predicate = {}
        if obs is None:
            pos_predicate['partner_zone'] = Direction.NORTH
            return pos_predicate
        else:
            pos_predicate['partner_zone'] = self._get_partner_zone(obs, player_id, partner_id)
            # pos_predicate['partner_zone'] = pos_predicate['partner_zone'].value
            return pos_predicate

    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        obs = obs[1]

        # Temporary sources (coordination)
        temporary_sources = self._get_temporary_sources(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(self._get_held_predicate(obs, 0))
        predicate.update(self._get_pot_state_predicate(obs))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='onion')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='tomato')
        )
        predicate.update(self._get_soup_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='dish')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='service')
        )
        predicate.update(self._get_pot_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(self._get_partner_zone_predicate(obs=obs, player_id=0, partner_id=1))

        if to_dict:
            return predicate
        else:
            return tuple(predicate[k] for k in predicate)

    def dict_to_tuple(self, predicates: dict):
        if isinstance(predicates, tuple): return predicates
        preds = [predicates['held 0']]
        preds.extend(predicates[f'pot {pos[0]} state'] for pos in self.sources['pot'])
        preds.extend((predicates['onion'], predicates['tomato'], predicates['soup'],
                      predicates['dish'], predicates['service']))
        preds.extend(predicates[f'pot {pos[0]} pos'] for pos in self.sources['pot'])
        preds.append(predicates['partner_zone'])

        return tuple(preds)

    def tuple_to_dict(self, predicates: tuple):
        if isinstance(predicates, dict): return predicates
        preds = {'held 0': predicates[0]}
        preds.update({f'pot {pos[0]} state': predicates[1+i] for i, pos in enumerate(self.sources['pot'])})
        preds.update({
            'onion':   predicates[1+len(self.sources['pot'])+0],
            'tomato':  predicates[1+len(self.sources['pot'])+1],
            'soup':    predicates[1+len(self.sources['pot'])+2],
            'dish':    predicates[1+len(self.sources['pot'])+3],
            'service': predicates[1+len(self.sources['pot'])+4]
        })
        preds.update({f'pot {pos[0]} pos': predicates[1+len(self.sources['pot'])+5+i]
                      for i, pos in enumerate(self.sources['pot'])})
        preds.update({'partner_zone': predicates[1+len(self.sources['pot'])+5+len(self.sources['pot'])]})
        return preds


class Discretizer14(Discretizer13):
    """Discretizer

    Predicates:
        - held
        - held_partner
        - pot_state
        - predicate_pos
        - partner_zone
    """

    def __init__(self, env: OrderEnforcing):
        """
        :param env: Environment
        """
        super().__init__(env)
        self.predicate_space.update({'held 1': Held})

    def discretize(self, obs: Tuple[Any, OvercookedState], to_dict=False):
        obs = obs[1]

        # Temporary sources (coordination)
        temporary_sources = self._get_temporary_sources(obs)

        # Build state
        # IMPORTANT!!! We have to add each predicate in the same order as the attribute self.PREDICATES !!!!
        predicate = {}
        predicate.update(self._get_held_predicate(obs, 0))
        predicate.update(self._get_held_predicate(obs, 1))
        predicate.update(self._get_pot_state_predicate(obs))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='onion')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='tomato')
        )
        predicate.update(self._get_soup_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='dish')
        )
        predicate.update(
            self._get_object_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0, object='service')
        )
        predicate.update(self._get_pot_pos_predicate(obs=obs, temporary_sources=temporary_sources, player_id=0))
        predicate.update(self._get_partner_zone_predicate(obs=obs, player_id=0, partner_id=1))

        if to_dict:
            return predicate
        else:
            return tuple(predicate[k] for k in predicate)

    def dict_to_tuple(self, predicates: dict):
        if isinstance(predicates, tuple): return predicates
        preds = [predicates['held 0'], predicates['held 1']]
        preds.extend(predicates[f'pot {pos[0]} state'] for pos in self.sources['pot'])
        preds.extend((predicates['onion'], predicates['tomato'], predicates['soup'],
                      predicates['dish'], predicates['service']))
        preds.extend(predicates[f'pot {pos[0]} pos'] for pos in self.sources['pot'])
        preds.append(predicates['partner_zone'])

        return tuple(preds)

    def tuple_to_dict(self, predicates: tuple):
        if isinstance(predicates, dict): return predicates
        preds = {'held 0': predicates[0], 'held 1': predicates[1]}
        preds.update({f'pot {pos[0]} state': predicates[2+i] for i, pos in enumerate(self.sources['pot'])})
        preds.update({
            'onion':   predicates[2+len(self.sources['pot'])+0],
            'tomato':  predicates[2+len(self.sources['pot'])+1],
            'soup':    predicates[2+len(self.sources['pot'])+2],
            'dish':    predicates[2+len(self.sources['pot'])+3],
            'service': predicates[2+len(self.sources['pot'])+4]
        })
        preds.update({f'pot {pos[0]} pos': predicates[2+len(self.sources['pot'])+5+i]
                      for i, pos in enumerate(self.sources['pot'])})
        preds.update({'partner_zone': predicates[2+len(self.sources['pot'])+5+len(self.sources['pot'])]})
        return preds