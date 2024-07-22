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

"""A pyspiel config for meta-generated fruit trading games.
"""

import collections

from ml_collections import config_dict

# from open_spiel.python.games.chat_games.envs.base_envs import trade_fruit_with_tone_info as env_trade_fruit_with_tone_info
from open_spiel.python.games.chat_games.envs.base_envs import letter as env_letter
from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.observations import utils as obs_utils
# from open_spiel.python.games.chat_games.envs.payoffs import trade_fruit as payoffs_trade_fruit
from open_spiel.python.games.chat_games.envs.payoffs import letter as payoffs_letter
# from open_spiel.python.games.chat_games.envs.scenarios.domains import trade_fruit as scenario_trade_fruit
from open_spiel.python.games.chat_games.envs.scenarios.domains import letter as scenario_letter
# from open_spiel.python.games.chat_games.envs.scenarios.players import names as names_trade_fruit
from open_spiel.python.games.chat_games.envs.scenarios.players import names as names_letter


def get_config():
  """Get configuration for chat game."""
  config = config_dict.ConfigDict()

  num_players = 2

  observations = [
      obs_utils.Observation(summary.PREFIX, summary.POSTFIX)
      for _ in range(num_players)
  ]

#   header = env_trade_fruit_with_tone_info.HEADER
  header = env_letter.HEADER

#   payoffs = [payoffs_trade_fruit.PAYOFF]
  payoffs = [payoffs_letter.PAYOFF]

#   examples_names = names_trade_fruit.NAMES
  examples_names = names_letter.NAMES

  given_prompt_actions = collections.OrderedDict()
  tones = ['calm',
           'assertive',
           'submissive',
           'any']
  given_prompt_actions[header.action_keys[0]] = tones
  num_tones = len(tones)

  examples_private_info = collections.OrderedDict()
#   examples_private_info['fruit_endowment'] = [scenario_trade_fruit.ENDOWMENT_A,
#                                               scenario_trade_fruit.ENDOWMENT_B,
#                                               scenario_trade_fruit.ENDOWMENT_C,
#                                               scenario_trade_fruit.ENDOWMENT_D]
#   examples_private_info['fruit_valuations'] = [scenario_trade_fruit.VALUATION_A,
#                                                scenario_trade_fruit.VALUATION_B,
#                                                scenario_trade_fruit.VALUATION_C,
#                                                scenario_trade_fruit.VALUATION_D]
  examples_private_info['quality'] = [scenario_letter.QUALITY_A,
                                      scenario_letter.QUALITY_B,
                                      scenario_letter.QUALITY_C,
                                      scenario_letter.QUALITY_D]
  examples_private_info['name'] = [scenario_letter.NAME_A,
                                   scenario_letter.NAME_B,
                                   scenario_letter.NAME_C,
                                   scenario_letter.NAME_D]
  examples_private_info['major'] = [scenario_letter.MAJOR_A,
                                    scenario_letter.MAJOR_B,
                                    scenario_letter.MAJOR_C,
                                    scenario_letter.MAJOR_D]
  examples_private_info['GPA'] = [scenario_letter.GPA_A,
                                    scenario_letter.GPA_B,
                                    scenario_letter.GPA_C,
                                    scenario_letter.GPA_D]
  examples_private_info['project'] = [scenario_letter.PROJECT_A,
                                        scenario_letter.PROJECT_B,
                                        scenario_letter.PROJECT_C,
                                        scenario_letter.PROJECT_D]
  examples_private_info['work_experience'] = [scenario_letter.WORK_EXPERIENCE_A,
                                                scenario_letter.WORK_EXPERIENCE_B,
                                                scenario_letter.WORK_EXPERIENCE_C,
                                                scenario_letter.WORK_EXPERIENCE_D]
  examples_private_info['academic'] = [scenario_letter.ACADEMIC_ACHIEVEMENTS_A,
                                                    scenario_letter.ACADEMIC_ACHIEVEMENTS_B,
                                                    scenario_letter.ACADEMIC_ACHIEVEMENTS_C,
                                                    scenario_letter.ACADEMIC_ACHIEVEMENTS_D]

#   scenario_a = env_trade_fruit_with_tone_info.Scenario(
#       scenario_trade_fruit.SCENARIO_A,
#       'Bob',
#       'Suzy',
#       scenario_trade_fruit.ENDOWMENT_A,
#       scenario_trade_fruit.VALUATION_A,
#       'calm')
#   scenario_b = env_trade_fruit_with_tone_info.Scenario(
#       scenario_trade_fruit.SCENARIO_B,
#       'Jill',
#       'George',
#       scenario_trade_fruit.ENDOWMENT_B,
#       scenario_trade_fruit.VALUATION_B,
#       'calm')

  scenario_a = env_letter.Scenario(
    msg='',
    sender='Alicia',
    receiver='Joel',
    quality=scenario_letter.QUALITY_A,
    name=scenario_letter.NAME_A,
    major=scenario_letter.MAJOR_A,
    GPA=scenario_letter.GPA_A,
    project=scenario_letter.PROJECT_A,
    work_experience=scenario_letter.WORK_EXPERIENCE_A,
    academic=scenario_letter.ACADEMIC_ACHIEVEMENTS_A)
  
  scenario_b = env_letter.Scenario(
    msg='',
    sender='Taylor',
    receiver='Marcus',
    quality=scenario_letter.QUALITY_B,
    name=scenario_letter.NAME_B,
    major=scenario_letter.MAJOR_B,
    GPA=scenario_letter.GPA_B,
    project=scenario_letter.PROJECT_B,
    work_experience=scenario_letter.WORK_EXPERIENCE_B,
    academic=scenario_letter.ACADEMIC_ACHIEVEMENTS_B)

  examples_scenarios = [scenario_a, scenario_b]

#   llm_termination_prompt = scenario_trade_fruit.LLM_TERMINATION_PROMPT
  llm_termination_prompt = scenario_letter.LLM_TERMINATION_PROMPT

  params = {'num_distinct_actions': num_players * num_tones,
            'num_llm_seeds': 2,
            'num_players': num_players,
            'min_utility': min([float(p.min) for p in payoffs]),
            'max_utility': max([float(p.max) for p in payoffs]),
            'num_max_replies': 2}

  config.params = params

  config.game = config_dict.ConfigDict()
  config.game.observations = observations
  config.game.header = header
  config.game.payoffs = payoffs
  config.game.given_prompt_actions = given_prompt_actions
  config.game.num_names = 10
  config.game.num_prompt_actions = (num_tones,)
  config.game.num_private_info = (3, 3, 3, 3, 3, 3, 3)
  config.game.examples_names = examples_names
  config.game.examples_private_info = examples_private_info
  config.game.examples_scenarios = examples_scenarios
  config.game.llm_list_suffix = 'Output: '
  config.game.llm_termination_prompt = llm_termination_prompt

  return config
