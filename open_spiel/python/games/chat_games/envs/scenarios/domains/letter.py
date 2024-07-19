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

"""Examples of recommendation letter scenarios -- useful for generating more examples.
"""

from open_spiel.python.games.chat_games.envs.observations import summary
from open_spiel.python.games.chat_games.envs.termination import utils as term_utils
from open_spiel.python.games.chat_games.envs.utils import text

# Scenario A
# SCENARIO_A_LIST = ['Dear {receiver},',
#                    'I am writing to recommend John Doe for [the position/program/scholarship] at your esteemed institution. John is a Computer Science major with a GPA of 3.8 out of 4.0 and has consistently demonstrated exceptional ability and dedication throughout his academic career.',
#                    'John has been actively involved in several impressive projects. Notably, he developed a machine learning model for predicting stock prices, showcasing his strong analytical skills and ability to apply theoretical knowledge to practical problems. Additionally, he created a mobile app for scheduling tasks, which highlights his versatility and commitment to improving user experience through technology.',
#                    'John\'s practical experience extends beyond the classroom. He completed an internship at Nvidia as a software developer, where he honed his skills in a professional setting and contributed to significant projects. This experience has not only enhanced his technical abilities but also provided him with valuable industry insights.',
#                    'Academically, John has made substantial contributions to the field of artificial intelligence. He co-authored a paper that was accepted at a top AI conference, demonstrating his research capabilities and his ability to collaborate effectively with peers and mentors.',
#                    'In summary, John Doe is a highly capable and motivated individual with a strong academic background, practical experience, and a proven track record in research. I am confident that he will excel in [the position/program/scholarship] and contribute significantly to your institution. I wholeheartedly recommend him without reservation.',
#                    'Please feel free to contact me if you need any further information.',
#                    'Sincerely,', '{sender}']
# SCENARIO_A = '\n\n'.join(text.wrap(SCENARIO_A_LIST))

QUALITY_A_LIST = ['Good']
QUALITY_A = '\n'.join(text.wrap(QUALITY_A_LIST))

NAME_A_LIST = ['John Doe']
NAME_A = '\n'.join(text.wrap(NAME_A_LIST))

MAJOR_A_LIST = ['Computer Science']
MAJOR_A = '\n'.join(text.wrap(MAJOR_A_LIST))

GPA_A_LIST = ['3.8/4.0']
GPA_A = '\n'.join(text.wrap(GPA_A_LIST))

PROJECT_A_LIST = ['Machine learning model for predicting stock prices', 'Mobile app for scheduling tasks']
PROJECT_A = '\n'.join(text.wrap(PROJECT_A_LIST))

WORK_EXPERIENCE_A_LIST = ['Internship at Nvidia as a software developer']
WORK_EXPERIENCE_A = '\n'.join(text.wrap(WORK_EXPERIENCE_A_LIST))

ACADEMIC_ACHIEVEMENTS_A_LIST = ['Co-authored a paper accepted at a top AI conference']
ACADEMIC_ACHIEVEMENTS_A = '\n'.join(text.wrap(ACADEMIC_ACHIEVEMENTS_A_LIST))


# Scenario B
# SCENARIO_B_LIST = ['Dear {receiver},',
#                    'I am pleased to recommend Mark Smith for the Master program at your esteemed institution. Mark is a Business Administration major, and I have had the pleasure of observing his growth and development over the past few years.',
#                    'Mark has achieved a GPA of 3.5 out of 4.0, reflecting his strong academic performance and commitment to his studies. He has consistently demonstrated a keen interest in business concepts and has excelled in his coursework.',
#                    'One of Mark\'s notable projects was the creation of an innovative business plan for a hypothetical retail store. This project showcased his ability to think strategically and apply business theories to real-world scenarios. His work was well-received, and he presented it at our university\'s annual business showcase.',
#                    'In addition to his academic achievements, Mark has co-authored a paper on retail management strategies, which was published in a reputable business journal. This publication highlights his research capabilities and his ability to contribute valuable insights to the field of business administration.',
#                    'Mark\'s dedication to his education is further exemplified by his active involvement in various extracurricular activities. He has been a member of our university\'s business club, where he has taken on leadership roles and organized events that have benefited his peers.',
#                    'I am confident that Mark Smith will excel in the Master program and make significant contributions to your institution. His academic achievements, research experience, and extracurricular involvement make him a well-rounded candidate who is ready to take on new challenges.',
#                    'Please feel free to contact me if you need any further information.',
#                    'Sincerely,', '{sender}']
# SCENARIO_B = '\n\n'.join(text.wrap(SCENARIO_B_LIST))

QUALITY_B_LIST = ['Bad']
QUALITY_B = '\n'.join(text.wrap(QUALITY_B_LIST))

NAME_B_LIST = ['Mark Smith']
NAME_B = '\n'.join(text.wrap(NAME_B_LIST))

MAJOR_B_LIST = ['Business Administration']
MAJOR_B = '\n'.join(text.wrap(MAJOR_B_LIST))

GPA_B_LIST = ['2.4/4.0']
GPA_B = '\n'.join(text.wrap(GPA_B_LIST))

PROJECT_B_LIST = ['Created a basic business plan for a hypothetical retail store']
PROJECT_B = '\n'.join(text.wrap(PROJECT_B_LIST))

WORK_EXPERIENCE_B_LIST = ['None']
WORK_EXPERIENCE_B = '\n'.join(text.wrap(WORK_EXPERIENCE_B_LIST))

ACADEMIC_ACHIEVEMENTS_B_LIST = ['Failed multiple core courses', 'No publications']
ACADEMIC_ACHIEVEMENTS_B = '\n'.join(text.wrap(ACADEMIC_ACHIEVEMENTS_B_LIST))

# Scenario C
# SCENARIO_C_LIST = ['Dear {receiver},',
#                    'I am writing to provide some insights on Emily Davis, who is applying for the PhD program in Biology at your esteemed institution. Emily is a Biology major at [Your University], and I have had the opportunity to observe her academic journey over the past few years.',
#                    'Emily has maintained a GPA of 3.2 out of 4.0, reflecting a solid but not exceptional academic performance. Her academic record indicates her persistent effort and dedication to her studies, despite facing various challenges.',
#                    'One of Emily\'s notable projects involved researching the effects of climate change on marine ecosystems. While the project was ambitious, it faced several hurdles, and the outcomes were not as impactful as initially anticipated. Nevertheless, her perseverance in addressing complex environmental issues is commendable.',
#                    'Emily also developed a community-based wildlife conservation program, which aimed to engage local communities in conservation efforts. The program was well-intentioned, although it struggled to achieve significant traction and long-term impact.',
#                    'Her practical experience includes an internship at a local marine biology research institute, where she gained exposure to foundational research techniques. This experience allowed her to understand the practical aspects of marine research, albeit at a basic level.',
#                    'Academically, Emily co-authored a paper that was published in a regional biology journal. While the publication is a noteworthy achievement, it did not receive significant attention within the broader scientific community. Nonetheless, it demonstrates her ability to contribute to academic research.',
#                    'In summary, Emily Davis possesses a solid foundation in biology and has shown a genuine interest in environmental issues. While her academic and research experiences have been mixed, she has demonstrated a willingness to learn and grow. I believe that with the right support and guidance, she can continue to develop her skills and make meaningful contributions to the field.',
#                    'Please feel free to contact me if you need any further information or have any questions.',
#                    'Sincerely,', '{sender}']
# SCENARIO_C = '\n\n'.join(text.wrap(SCENARIO_C_LIST))

QUALITY_C_LIST = ['Good']
QUALITY_C = '\n'.join(text.wrap(QUALITY_C_LIST))

NAME_C_LIST = ['Emily Davis']
NAME_C = '\n'.join(text.wrap(NAME_C_LIST))

MAJOR_C_LIST = ['Biology']
MAJOR_C = '\n'.join(text.wrap(MAJOR_C_LIST))

GPA_C_LIST = ['3.7/4.0']
GPA_C = '\n'.join(text.wrap(GPA_C_LIST))

PROJECT_C_LIST = ['Conducted research on the effects of climate change on marine ecosystems', 'Developed a community-based wildlife conservation program']
PROJECT_C = '\n'.join(text.wrap(PROJECT_C_LIST))

WORK_EXPERIENCE_C_LIST = ['Internship at a marine biology research institute']
WORK_EXPERIENCE_C = '\n'.join(text.wrap(WORK_EXPERIENCE_C_LIST))

ACADEMIC_ACHIEVEMENTS_C_LIST = ['Co-authored a paper published in a leading biology journal']
ACADEMIC_ACHIEVEMENTS_C = '\n'.join(text.wrap(ACADEMIC_ACHIEVEMENTS_C_LIST))

# Scenario D
# SCENARIO_D_LIST = ['Dear {receiver},',
#                    'I am writing to provide some insights on Michael Green, who is applying for the Master’s program in History at your esteemed institution. Michael is a History major at [Your University], and I have had the opportunity to observe his academic journey over the past few years.',
#                    'Michael has maintained a GPA of 2.3 out of 4.0, which reflects the various academic challenges he has encountered. Despite these challenges, Michael has shown a consistent effort to engage with his studies and improve his understanding of historical subjects.',
#                    'One of Michael\'s projects involved writing an essay on the French Revolution. While the essay provided a basic overview of the historical event, it demonstrated Michael\'s interest in exploring significant historical periods. Although the depth of analysis in the essay was limited, it was a valuable exercise in developing his research and writing skills.',
#                    'Michael has not yet gained formal work experience, which has limited his exposure to practical applications of his historical knowledge. However, his academic experiences have provided him with a theoretical foundation in history.',
#                    'Academically, Michael has faced difficulties in major courses and has not yet published any academic work. These experiences highlight areas where he can further develop his skills and knowledge. With additional support and resources, I believe Michael has the potential to grow academically and make progress in his studies.',
#                    'In summary, Michael Green has shown a persistent interest in history and a willingness to overcome academic obstacles. While his academic record has been mixed, he has demonstrated a commitment to his education. I believe that with the right support and guidance, he can continue to develop his skills and pursue his interest in history at the graduate level.',
#                    'Please feel free to contact me if you need any further information or have any questions.',
#                    'Sincerely,', '{sender}']
# SCENARIO_D = '\n\n'.join(text.wrap(SCENARIO_D_LIST))

QUALITY_D_LIST = ['Bad']
QUALITY_D = '\n'.join(text.wrap(QUALITY_D_LIST))

NAME_D_LIST = ['Michael Green']
NAME_D = '\n'.join(text.wrap(NAME_D_LIST))

MAJOR_D_LIST = ['History']
MAJOR_D = '\n'.join(text.wrap(MAJOR_D_LIST))

GPA_D_LIST = ['2.3/4.0']
GPA_D = '\n'.join(text.wrap(GPA_D_LIST))

PROJECT_D_LIST = ['Wrote a basic essay on the French Revolution']
PROJECT_D = '\n'.join(text.wrap(PROJECT_D_LIST))

WORK_EXPERIENCE_D_LIST = ['None']
WORK_EXPERIENCE_D = '\n'.join(text.wrap(WORK_EXPERIENCE_D_LIST))

ACADEMIC_ACHIEVEMENTS_D_LIST = ['Poor performance in major courses', 'No publications']
ACADEMIC_ACHIEVEMENTS_D = '\n'.join(text.wrap(ACADEMIC_ACHIEVEMENTS_D_LIST))

query = ('Read the following summary of a dialgoue between two parties ' +
         'attempting to reach a hiring agreement. Have the players reached a ' +
         'hiring agreement? If the student has been accepted or the players cannot' +
         ' give an offer, respond Yes. Otherwise, if the players are ' +
         'still discussing terms, respond No.' +
         'Here is the dialogue:\n\n{msg}\n\n' + '&' *50 +
         'Response: ')

LLM_TERMINATION_PROMPT = term_utils.Termination(query,
                                                summary.PREFIX,
                                                summary.POSTFIX)
