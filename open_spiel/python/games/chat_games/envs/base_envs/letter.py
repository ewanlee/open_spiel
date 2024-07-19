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

"""A base environment for trading fruit with private info.
"""

import dataclasses

# from open_spiel.python.games.chat_games.envs.comm_substrates import trades
from open_spiel.python.games.chat_games.envs.comm_substrates import letters
# from open_spiel.python.games.chat_games.envs.scenarios.domains import trade_fruit
from open_spiel.python.games.chat_games.envs.scenarios.domains import letter
from open_spiel.python.games.chat_games.envs.utils import header
from open_spiel.python.games.chat_games.envs.utils import text


action_keys = tuple(['tone'])
# info_keys = tuple(['fruit_endowment', 'fruit_valuations'])
info_keys = tuple(['quality', 'name', 'major', 'GPA', 'project', 'work_experience', 'academic'])

# w_opts = (trades.W_OPTS_PREFIX +
#           'Fruit Endowment:\n{fruit_endowment}\n\n' +
#           'Fruit Valuations:\n{fruit_valuations}\n\n' +
#           'Tone: Use a {tone} tone.\n' +
#           trades.PLAIN)

w_opts = (letters.W_OPTS_PREFIX +
          'Quality: {quality}\n\n' +
          'Name: {name}\n\n' +
          'Major: {major}\n\n' +
          'GPA: {GPA}\n\n' +
          'Project: {project}\n\n' +
          'Work Experience: {work_experience}\n\n' +
          'Academic: {academic}\n\n' +
          # 'Tone: Use a {tone} tone.\n' +
          letters.PLAIN)

# Example a

# email_1a = ['Hi Joel,',
#             'I would like to trade you 2 strawberries for 3 blueberries.',
#             'Would you like to trade with me?',
#             'Best,', 'Bob']
# email_1a = (trades.PLAIN.format(sender='Alicia', receiver='Joel') +
#             '\n\n'.join(text.wrap(email_1a)))
letter_a = ['Dear {receiver},',
                   'I am writing to recommend John Doe for [the position/program/scholarship] at your esteemed institution. John is a Computer Science major with a GPA of 3.8 out of 4.0 and has consistently demonstrated exceptional ability and dedication throughout his academic career.',
                   'John has been actively involved in several impressive projects. Notably, he developed a machine learning model for predicting stock prices, showcasing his strong analytical skills and ability to apply theoretical knowledge to practical problems. Additionally, he created a mobile app for scheduling tasks, which highlights his versatility and commitment to improving user experience through technology.',
                   'John\'s practical experience extends beyond the classroom. He completed an internship at Nvidia as a software developer, where he honed his skills in a professional setting and contributed to significant projects. This experience has not only enhanced his technical abilities but also provided him with valuable industry insights.',
                   'Academically, John has made substantial contributions to the field of artificial intelligence. He co-authored a paper that was accepted at a top AI conference, demonstrating his research capabilities and his ability to collaborate effectively with peers and mentors.',
                   'In summary, John Doe is a highly capable and motivated individual with a strong academic background, practical experience, and a proven track record in research. I am confident that he will excel in [the position/program/scholarship] and contribute significantly to your institution. I wholeheartedly recommend him without reservation.',
                   'Please feel free to contact me if you need any further information.',
                   'Sincerely,', '{sender}']
letter_a = '\n\n'.join(text.wrap(letter_a)).format(sender='Alicia', receiver='Joel')
# letter_a = (letters.PLAIN.format(sender='Alicia', receiver='Joel') + letter_a)

# email_2a = ['Hi Alicia,',
#             'Thanks for reaching out. I only have 2 blueberries, but even if ' +
#             'I had 3, I would not want to give them up. Also, I dislike ' +
#             'strawberries. I do not think a trade makes sense in this case.',
#             'Thanks for considering trading with me though!',
#             'Best,', 'Joel']
# email_2a = (trades.PLAIN.format(sender='Joel', receiver='Alicia') +
#             '\n\n'.join(text.wrap(email_2a)))

# email_3a = ['Hi Joel,',
#             'That is all well. I understand.',
#             'Have a good day!',
#             'Best,', 'Alicia']
# email_3a = (trades.PLAIN.format(sender='Alicia', receiver='Joel') +
#             '\n\n'.join(text.wrap(email_3a)))

# example_a = email_1a + email_2a
# example_a = example_a.strip('\n')

# Example b

# email_1b = ['Hi Marcus,',
#             'I would like to trade you 2 kiwis for 1 watermelon.',
#             'Would you like to trade with me?',
#             'Best,', 'Taylor']
# email_1b = (trades.PLAIN.format(sender='Taylor', receiver='Marcus') +
#             '\n\n'.join(text.wrap(email_1b)))

letter_b = ['Dear {receiver},',
                   'I am pleased to recommend Mark Smith for the Master program at your esteemed institution. Mark is a Business Administration major, and I have had the pleasure of observing his growth and development over the past few years.',
                   'Mark has achieved a GPA of 3.5 out of 4.0, reflecting his strong academic performance and commitment to his studies. He has consistently demonstrated a keen interest in business concepts and has excelled in his coursework.',
                   'One of Mark\'s notable projects was the creation of an innovative business plan for a hypothetical retail store. This project showcased his ability to think strategically and apply business theories to real-world scenarios. His work was well-received, and he presented it at our university\'s annual business showcase.',
                   'In addition to his academic achievements, Mark has co-authored a paper on retail management strategies, which was published in a reputable business journal. This publication highlights his research capabilities and his ability to contribute valuable insights to the field of business administration.',
                   'Mark\'s dedication to his education is further exemplified by his active involvement in various extracurricular activities. He has been a member of our university\'s business club, where he has taken on leadership roles and organized events that have benefited his peers.',
                   'I am confident that Mark Smith will excel in the Master program and make significant contributions to your institution. His academic achievements, research experience, and extracurricular involvement make him a well-rounded candidate who is ready to take on new challenges.',
                   'Please feel free to contact me if you need any further information.',
                   'Sincerely,', '{sender}']
letter_b = '\n\n'.join(text.wrap(letter_b)).format(sender='Taylor', receiver='Marcus')
# letter_b = (letters.PLAIN.format(sender='Taylor', receiver='Marcus') + letter_b)

# email_2b = ['Hi Taylor,',
#             'I love kiwis! And lucky for you, I have a watermelon.',
#             'Lets trade!',
#             'Best,', 'Marcus']
# email_2b = (trades.PLAIN.format(sender='Marcus', receiver='Taylor') +
#             '\n\n'.join(text.wrap(email_2b)))

# email_3b = ['Hi Marcus,',
#             'Great! It was a pleasure negotiating with you.',
#             'Have a good day!',
#             'Best,', 'Taylor']
# email_3b = (trades.PLAIN.format(sender='Taylor', receiver='Marcus') +
#             '\n\n'.join(text.wrap(email_3b)))

# example_b = email_1b + email_2b + email_3b
# example_b = example_b.strip('\n')

# Example c

# email_1c = ['Hi Suzy,',
#             'I would like to trade you 1 banana for 1 apple.',
#             'Would you like to trade with me?',
#             'Best,', 'Bob']
# email_1c = (trades.PLAIN.format(sender='Bob', receiver='Suzy') +
#             '\n\n'.join(text.wrap(email_1c)))

letter_c = ['Dear {receiver},',
                   'I am writing to provide some insights on Emily Davis, who is applying for the PhD program in Biology at your esteemed institution. Emily is a Biology major at [Your University], and I have had the opportunity to observe her academic journey over the past few years.',
                   'Emily has maintained a GPA of 3.2 out of 4.0, reflecting a solid but not exceptional academic performance. Her academic record indicates her persistent effort and dedication to her studies, despite facing various challenges.',
                   'One of Emily\'s notable projects involved researching the effects of climate change on marine ecosystems. While the project was ambitious, it faced several hurdles, and the outcomes were not as impactful as initially anticipated. Nevertheless, her perseverance in addressing complex environmental issues is commendable.',
                   'Emily also developed a community-based wildlife conservation program, which aimed to engage local communities in conservation efforts. The program was well-intentioned, although it struggled to achieve significant traction and long-term impact.',
                   'Her practical experience includes an internship at a local marine biology research institute, where she gained exposure to foundational research techniques. This experience allowed her to understand the practical aspects of marine research, albeit at a basic level.',
                   'Academically, Emily co-authored a paper that was published in a regional biology journal. While the publication is a noteworthy achievement, it did not receive significant attention within the broader scientific community. Nonetheless, it demonstrates her ability to contribute to academic research.',
                   'In summary, Emily Davis possesses a solid foundation in biology and has shown a genuine interest in environmental issues. While her academic and research experiences have been mixed, she has demonstrated a willingness to learn and grow. I believe that with the right support and guidance, she can continue to develop her skills and make meaningful contributions to the field.',
                   'Please feel free to contact me if you need any further information or have any questions.',
                   'Sincerely,', '{sender}']
letter_c = '\n\n'.join(text.wrap(letter_c)).format(sender='Bob', receiver='Suzy')
# letter_c = (letters.PLAIN.format(sender='Bob', receiver='Suzy') + letter_c)

# email_2c = ['Hi Bob,',
#             'Thanks for reaching out. I really like my apples so I am ' +
#             'hesitant to give them up. Would you be willing to take a few ' +
#             'kiwis instead? I would like to trade you 3 kiwis for 1 banana.',
#             'Does that work?',
#             'Best,', 'Suzy']
# email_2c = (trades.PLAIN.format(sender='Suzy', receiver='Bob') +
#             '\n\n'.join(text.wrap(email_2c)))

# email_3c = ['Hi Suzy,',
#             'Yes! I would have preferred an apple but 3 kiwis are nearly as ' +
#             'good and I would rather have those than a banana.',
#             'Thanks for trading with me!',
#             'Best,', 'Bob']
# email_3c = '\n\n'.join(text.wrap(email_3c))

# example_c = email_1c + email_2c
# example_c = example_c.strip('\n')

# Example d

letter_d = ['Dear {receiver},',
                   'I am writing to provide some insights on Michael Green, who is applying for the Master’s program in History at your esteemed institution. Michael is a History major at [Your University], and I have had the opportunity to observe his academic journey over the past few years.',
                   'Michael has maintained a GPA of 2.3 out of 4.0, which reflects the various academic challenges he has encountered. Despite these challenges, Michael has shown a consistent effort to engage with his studies and improve his understanding of historical subjects.',
                   'One of Michael\'s projects involved writing an essay on the French Revolution. While the essay provided a basic overview of the historical event, it demonstrated Michael\'s interest in exploring significant historical periods. Although the depth of analysis in the essay was limited, it was a valuable exercise in developing his research and writing skills.',
                   'Michael has not yet gained formal work experience, which has limited his exposure to practical applications of his historical knowledge. However, his academic experiences have provided him with a theoretical foundation in history.',
                   'Academically, Michael has faced difficulties in major courses and has not yet published any academic work. These experiences highlight areas where he can further develop his skills and knowledge. With additional support and resources, I believe Michael has the potential to grow academically and make progress in his studies.',
                   'In summary, Michael Green has shown a persistent interest in history and a willingness to overcome academic obstacles. While his academic record has been mixed, he has demonstrated a commitment to his education. I believe that with the right support and guidance, he can continue to develop his skills and pursue his interest in history at the graduate level.',
                   'Please feel free to contact me if you need any further information or have any questions.',
                   'Sincerely,', '{sender}']
letter_d = '\n\n'.join(text.wrap(letter_d)).format(sender='Rookie', receiver='TheShy')

instr_a = ['You are an assistant who is playing a game where you trade fruit.' +
           ' You want to make a trade that is best for you. You will read a ' +
           'dialogue that contains a conversation where you have been ' +
           'negotiating to trade your fruit for another persons fruit. You ' +
           'will then read a text block that contains information a) about ' +
           'the actual fruit you currently have and are able to trade and b)' +
           ' information about how much you value certain types of fruit.',
           'You should use everything you learned from this to decide to ',
           '1) accept the trade if you are happy with the terms,',
           '2) reject the negotiation all together and say goodbye if you do ' +
           'not think an agreement can be reached,',
           '3) counter-propose an alternative trade that includes what fruit ' +
           'you would like to give and what fruit you would like to receive ' +
           'in turn.',
           'Consider the following example dialogues. Components of the ' +
           'examples will be demarked with the symbol "&". Here is the first ' +
           'example which shows a trade is rejected.',
           '&' * 50]

instr_a = [
    "You are a senior professor, and students often come to you for recommendation letters. The quality of these students varies, but only you know their true information, which includes their overall quality, name, major, GPA, project experience, work experience, and academic experience. The readers of the recommendation letters are the HRs of a university, and they can know the true quality of the student after giving them an offer. They can also judge the student's quality based on their past experience, though not necessarily accurately. To maximize the chances of your recommended students being accepted, you can truthfully convey the student's information in the recommendation letter, or you can conceal, confuse, or even provide some false information about the student. For your long-term benefit, you don't necessarily have to recommend every student in the recommendation letter. You could recommend a bad student or a good student to maximize your long-term reward.",
    '&' * 50,
]


instr_b = ['&' * 50,
           'Here\'s the first example: recommend a good student and truthfully provide their real information in the recommendation letter.',
           '&' * 50]

instr_c = ['&' * 50,
           'Here\'s the second example: recommend a bad student by hiding, modifying, or providing false information.',
           '&' * 50]

instr_d = ['&' * 50,
           'Here\'s the third example: similarly, modify the information of a good student, such as GPA, research experience, etc., to avoid recommending this student.',
           '&' * 50]

instr_e = ['&' * 50,
           'This is the last example: do not recommend a bad student and truthfully provide their real information in the recommendation letter.',
           '&' * 50]

# info = w_opts.format(sender='Bob', receiver='Suzy',
#                      fruit_endowment=trade_fruit.ENDOWMENT_A,
#                      fruit_valuations=trade_fruit.VALUATION_A,
#                      tone='calm').strip('\n')

info_a = w_opts.format(sender='Alicia', receiver='Joel',
                       quality=letter.QUALITY_A,
                       name=letter.NAME_A,
                       major=letter.MAJOR_A,
                       GPA=letter.GPA_A,
                       project=letter.PROJECT_A,
                       work_experience=letter.WORK_EXPERIENCE_A,
                       academic=letter.ACADEMIC_ACHIEVEMENTS_A).strip('\n')

info_b = w_opts.format(sender='Taylor', receiver='Marcus',
                        quality=letter.QUALITY_B,
                        name=letter.NAME_B,
                        major=letter.MAJOR_B,
                        GPA=letter.GPA_B,
                        project=letter.PROJECT_B,
                        work_experience=letter.WORK_EXPERIENCE_B,
                        academic=letter.ACADEMIC_ACHIEVEMENTS_B).strip('\n')

info_c = w_opts.format(sender='Bob', receiver='Suzy',
                        quality=letter.QUALITY_C,
                        name=letter.NAME_C,
                        major=letter.MAJOR_C,
                        GPA=letter.GPA_C,
                        project=letter.PROJECT_C,
                        work_experience=letter.WORK_EXPERIENCE_C,
                        academic=letter.ACADEMIC_ACHIEVEMENTS_C).strip('\n')

info_d = w_opts.format(sender='Rookie', receiver='TheShy',
                        quality=letter.QUALITY_D,
                        name=letter.NAME_D,
                        major=letter.MAJOR_D,
                        GPA=letter.GPA_D,
                        project=letter.PROJECT_D,
                        work_experience=letter.WORK_EXPERIENCE_D,
                        academic=letter.ACADEMIC_ACHIEVEMENTS_D).strip('\n')

# instr_e = ['&' * 50,
#            'A reasonable way to respond would be as follows:',
#            '&' * 50]

# instr_f = ['&' * 50,
#            'Now you are going to read a fresh dialogue, fruit endowment, and ' +
#            'fruit valuation information. Please give a reasonable response ' +
#            'that attempts to reach an agreement to trade fruit.',
#            '&' * 50]

instr_f = ['&' * 50,
           'Now you are going to read a fresh student information, the overall quality, name, major, GPA, project experience, work experience, and academic experience. Please give a reasonable recommendation letter that attempts to reach an agreement with the HR.',
           '&' * 50]

# context = (text.wrap(instr_a) + [example_a] + text.wrap(instr_b) +[example_b] +
#            text.wrap(instr_c) + [example_c] + text.wrap(instr_d) + [info] +
#            text.wrap(instr_e) + [email_3c] + text.wrap(instr_f))

context = (text.wrap(instr_a) + 
           text.wrap(instr_b) + [info_a] + [letter_a] +
           text.wrap(instr_c) + [info_b] + [letter_b] +
           text.wrap(instr_d) + [info_c] + [letter_c] +
           text.wrap(instr_e) + [info_d] + [letter_d] + 
           text.wrap(instr_f))

# HEADER = header.Header(trades.PLAIN,
#                        w_opts,
#                        trades.strip_msg,
#                        trades.SPECIAL_CHARS,
#                        action_keys,
#                        info_keys,
#                        '\n\n'.join(context))

HEADER = header.Header(letters.PLAIN,
                        w_opts,
                        letters.strip_msg,
                        letters.SPECIAL_CHARS,
                        action_keys,
                        info_keys,
                        '\n\n'.join(context))

# @dataclasses.dataclass(frozen=True)
# class Scenario(header.BaseScenario):
#   fruit_endowment: str
#   fruit_valuations: str
#   tone: str = 'calm'

@dataclasses.dataclass(frozen=True)
class Scenario(header.BaseScenario):
  quality: str
  name: str
  major: str
  GPA: str
  project: str
  work_experience: str
  academic: str
