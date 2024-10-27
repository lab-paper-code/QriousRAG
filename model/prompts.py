PROMPT = {}

## eval docs
PROMPT['eval_doc_instr'] = '''
You are given an query and an exerpt of a document. If the document contains relevant information to answer the query output: #relevant\n 
If the document is not relevant output: #irrelevant\n.
Do not comment your output, but strictly follow these instructions.

Query: Who has the highest goals in world football?
Doc: in top - level association football competitions, 25 players have scored 500 or more goals in both club and international football, according to research by the iffhs, [ 1 ] first published in 2007. [ 2 ] taking into account competitions of all levels, 77 players have reached the milestone, according to research by the rsssf, [ 3 ] an organisation described by german newspaper der spiegel as a " wikipedia of football statistics ". [ 4 ] hungarian imre schlosser was the first to reach the 500 - goal mark, doing so in 1927 shortly before his retirement. [ 5 ] nine players have accomplished the feat at a single club : josef bican ( slavia prague ), jimmy jones ( glenavon ), jimmy mcgrory ( celtic ), joe bambrick ( linfield ), lionel messi ( barcelona ), gerd muller ( bayern munich ), pele ( santos ), fernando peyroteo ( sporting cp ), and uwe seeler ( hamburg ). [ 6 ] of these nine, messi scored the most, with 672 goals between his debut in 2004 and his departure in 2021. [ 7 ] fifa, the international governing body of football,
'''

PROMPT['eval_doc_answ1'] = '''#relevant\n''' #25 players scored 500 goals or more in both club and international football. Nine players achieved this feat at a single club, with Lionel Messi scoring the most with 672 goals.'''

PROMPT['eval_doc_ex2'] = '''Query: Who has the highest goals in world football?
Doc:  # table : sounds of silence studio album by simon & garfunkel releasedjanuary 17, 1966 ( 1966 - 01 - 17 ) recorded ; april 5 – december 22, 1965 ; ( 
'''

PROMPT['eval_doc_answ2'] = '''#irrelevant\n'''

PROMPT['eval_doc_ex3'] = '''Query: Who has the highest goals in world football?
Doc:  matches considered official internationals by the opposing sides, which would make him the first footballer to score 50 or more international goals, ahead of imre schlosser, and was the fastest to achieve the feat, scoring his 50th goal in his 32nd official international match, with a four - goal haul against hungary on 31 may 1909. [ 17 ] puskas overall scored 84 goals in his international career, [ 11 ] and remained the highest international goalscorer for 24 years following his 84th goal in 1956 against austria, until mokhtar dahari of malaysia broke the record in the merdeka tournament after scoring his 85th goal on 27 october 1980 against kuwait and he went on to score 89 goals for his country in 142 international appearances. [ 18 ] [ 19 ] [ 20 ] in 2004, ali daei of iran broke the record after scoring his 90th goal against lebanon. [ 21 ] [ 22 ] daei also became the first player to score over 100 goals in international football, ending his career with 108 in total. [ 23 ] [ 24 ] his 100th goal came on 17 november 2004, when he scored a four - goal haul against laos in a 2006 fifa world cup qualification match. [ 25 ] however,
'''

PROMPT['eval_doc_answ3'] = '''#relevant\n''' #Imre Schlosser, Ferenc Puskas, Mokhtar Dahari, Ali Daei, who all hold the record for most international goals. \n Ferenc Puskas was the first footballer to score 50 or more international goals. \n Ali Daei became the first player to score over 100 goals in international football.'''

# refine query
PROMPT['refine_query_instr'] = '''
    Instruction: You are given an ambiguous query and several excerpts of documents containing relevant information. 
    Given the query and the relevant documents, what kind of curious follow up questions should be posed to answer the inital query in the better way?.


    Original Query: Who has the highest goals in world football?
    Context information: ['25 players scored 500 goals or more in both club and international football. Nine players achieved this feat at a single club, with Lionel Messi scoring the most with 672 goals.',
    'Cristiano Ronaldo holds the all-time record with 133 international goals.',
    'The document contains information about the highest goals in world football, but it does not provide a direct answer to the question. However, it mentions that Just Fontaine holds the record for most consecutive matches scored with 6 goals.',
    'Ferenc Puskas, who scored 84 goals in his international career and remained the highest international goalscorer for 24 years. He was later surpassed by Mokhtar Dahari of Malaysia, who scored 89 goals, and then Ali Daei of Iran, who broke the record with 108 goals.']
    '''

PROMPT['refine_query_answ1'] = '''
    ['### Who are the 25 players who have scored 500 goals or more in world football?',
    '### Who currently holds the record for the most international goals in world football?',
    '### Who holds the record for most consecutive matches scored with goals in international football?',
    '### Who is the new highest international goalscorer and how many goals did he score?']
    '''

PROMPT['refine_query_ex2'] = '''
    Original Query: Who won the ncaa football national championship played in 2016?
    Context information: ['Alabama won the 2016 NCAA Football National Championship, defeating Georgia. The game was tied at the end of regulation and went into overtime, where Alabama scored a touchdown on second down to win the game.', 
    'The 2016 NCAA Football National Championship was played between Alabama and Clemson, with Alabama winning 24-7 in the semifinals and 31-0 in the championship game.']
    '''

PROMPT['refine_query_answ2'] = '''
    ['### Who were the teams that made it to the semifinals of the 2016 NCAA Football National Championship?',
    '### Which team won the championship game of the 2016 NCAA Football National Championship?',
    '### Which team won the semifinals of the 2016 NCAA Football National Championship?',
    '### Which team won the 2016 NCAA Football National Championship and how did they win?',
    '### Which team won the championship game in overtime in the 2016 NCAA Football National Championship?']
    '''

# make new answer
PROMPT['new_answer_instr']='''
    Instruction: You are given an query and an exerpt of a document. 
    Answer the query using context information.
    Strictly follow this instructions.
    
    Query: what place did the New York Jets finish in 2015?
    Context information: ['The New York Jets finished in 2nd place in the AFC East in the 2015 season, with a record of 10-6.', 
    'The New York Jets finished 1st in their 2015 season opener, winning against the Cleveland Browns with a score of 31-10.', 
    'The New York Jets finished 2nd in the AFC East in the 2015 season with a 10-6 record.']
'''

PROMPT['new_answer_answ1']='''
    The New York Jets finished in 2nd place in the AFC East during the 2015 season, with a record of 10-6. They maintained solid performance throughout the season, starting strong with a 31-10 victory against the Cleveland Browns in their season opener. However, despite their efforts, they ultimately ended up in 2nd place in the AFC East.
'''

PROMPT['new_answer_ex2']='''
    Query: Who won the ncaa football national championship played in 2016?
    Context information: ['Clemson won the NCAA Football National Championship in overtime in 2016, defeating Alabama.',
  'The score of the 2016 NCAA Football National Championship game between Alabama and Georgia was 35-31.',
  'The opposing teams for Alabama in the 2016 NCAA Football National Championship and the semifinals were Washington and Clemson.']
'''

PROMPT['new_answer_answ2']='''
    Clemson won the NCAA Football National Championship in 2016 by defeating Alabama in a thrilling overtime match. The game was highly competitive, with both teams putting up a strong fight, but Clemson ultimately came out on top. This victory marked a significant achievement for Clemson as they overcame a tough opponent, Alabama, in the championship game.
'''