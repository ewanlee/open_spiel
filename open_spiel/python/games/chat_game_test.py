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
      # dict(testcase_name='fixed_scenario', fixed_scenario=True),
      dict(testcase_name='random_scenario', fixed_scenario=False)
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

    pyspiel.random_sim_test(game, num_sims=1, serialize=False, verbose=True)


if __name__ == '__main__':
  absltest.main()
