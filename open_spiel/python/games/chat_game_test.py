# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pyspiel Chat Game."""

from absl.testing import absltest
from absl.testing import parameterized

from open_spiel.python.games import chat_game  # pylint: disable=unused-import

from open_spiel.python.games.chat_games.configs import config_fixed_mock
from open_spiel.python.games.chat_games.configs import config_trade_fruit_w_tone_fixed
from open_spiel.python.games.chat_games.configs import config_trade_fruit_w_tone
from open_spiel.python.games.chat_games.configs import config_rnd_mock

from open_spiel.python.games.chat_games.utils import test_utils as chat_test_utils

from open_spiel.python import policy as policy_lib

import pyspiel


# GLOBAL_TEST_LLM = chat_test_utils.TestLLM.MOCK
GLOBAL_TEST_LLM = chat_test_utils.TestLLM.LLAMA2CHAT


class ChatGameTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # self.fixed_config = config_fixed_mock.get_config()
    self.fixed_config = config_trade_fruit_w_tone_fixed.get_config()
    # self.random_config = config_rnd_mock.get_config()
    self.random_config = config_trade_fruit_w_tone.get_config()

    # vectorizer = chat_test_utils.MockVectorizer()
    vectorizer = chat_test_utils.Llama2ChatVectorizer()
    self.vectorize = vectorizer.vectorize

  @parameterized.named_parameters(
      dict(testcase_name='fixed_scenario', fixed_scenario=True),
      # dict(testcase_name='random_scenario', fixed_scenario=False)
  )
  def test_game_from_cc(self, fixed_scenario):
    """Runs our standard game tests, checking API consistency."""

    if fixed_scenario:
      config = self.fixed_config
    else:
      config = self.random_config

    game = pyspiel.load_game('chat_game', config.params.to_dict())

    game.load_chat_game(llm_type=GLOBAL_TEST_LLM,
                        vectorize=self.vectorize,
                        seed=1234,
                        **config.game)

    # pyspiel.random_sim_test(game, num_sims=1, serialize=False, verbose=True)

    print(game)

    # Some properties of the games.
    print("Number of players:", game.num_players())
    print("Maximum utility:", game.max_utility())
    print("Minimum utility:", game.min_utility())
    print("Number of distinct actions:", game.num_distinct_actions())

    # # Basic information about states.
    # print("Current player:", state.current_player())
    # print("Is terminal state:", state.is_terminal())
    # print("Returns:", state.returns())
    # print("Legal actions:", state.legal_actions())

    # Initialization: Bob proposes a trade to Suzy
    state = game.new_initial_state()
    print("Current player:", state.current_player())     # Get the ID of the current player (special chance player ID)
    print("Is chance node:", state.is_chance_node())     # Check if the current state is a chance node
    if state.is_chance_node():
      print("Chance outcomes:", state.chance_outcomes())    # Get the distribution over outcomes as a list of (outcome, probability) pairs
    print("Legal actions:", state.legal_actions())
    for action in state.legal_actions():
      print("Action:", state.action_to_string(action))
      # calm, assertive, subnmissive, any
    print("Information State:", state.information_state_string())

    # Take an action (Suzy)
    input("Before taking action...")
    state.apply_action(1)
    input("After taking action...")
    print("Current player:", state.current_player())
    print("Is chance node:", state.is_chance_node())
    if state.is_chance_node():
      # Get the distribution over outcomes as a list of (outcome, probability) pairs
      print("Chance outcomes:", state.chance_outcomes())    
      # Chance outcomes: [(0, 0.5), (1, 0.5)]
    
    # Take an action (Chance)
    input("Before taking action...")
    state.apply_action(1)
    input("After taking action...")
    print("Is chance node:", state.is_chance_node())
    print("Current state:", state)
    print("Current player:", state.current_player())
    print("Legal actions:", state.legal_actions())
    
    # Take an action (Bob)
    input("Before taking action...")
    state.apply_action(1)
    input("After taking action...")
    print("Is chance node:", state.is_chance_node())
    print("Current state:", state)
    print("Current player:", state.current_player())
    print("Legal actions:", state.legal_actions())

    # Take an action (Chance)
    input("Before taking action...")
    state.apply_action(1)
    input("After taking action...")
    print("Is chance node:", state.is_chance_node())
    print("Current state:", state)
    print("Current player:", state.current_player())
    print("Legal actions:", state.legal_actions())
    
    print("Is terminal state:", state.is_terminal())
    print("Returns:", state.returns())

    policy = policy_lib.UniformRandomPolicy(game)
    print("Action probability array:", policy.action_probabilities(game.new_initial_state()))

if __name__ == '__main__':
  absltest.main()
