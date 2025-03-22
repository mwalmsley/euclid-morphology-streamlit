import json
import logging
from copy import deepcopy

import streamlit as st
import numpy as np
import pandas as pd

def main(df):
    st.title('Euclid Explorer')
    st.subheader('by Mike Walmsley')

    st.markdown(
    """
    Euclid Q1 includes deep learning classifications for the visual appearance of all bright or extended galaxies. Our model learns from Galaxy Zoo volunteers and predicts what they would say for each galaxy.

    Explore the predictions using the filters on the left. Do you agree with the model?

    To read more about how the model works, check out the [Scaling Laws preprint](https://arxiv.org/abs/2404.02973) and the [Euclid preprint](https://arxiv.org/abs/2503.15310).
    """
    )
    interactive_galaxies(df)


class Questions():
        
        def __init__(self, branch):

            branches = {
                'smooth': ['smooth-or-featured', 'how-rounded', 'edge-on-bulge', 'merging'],
                'featured': ['smooth-or-featured', 'disk-edge-on', 'bar', 'has-spiral-arms', 'bulge-size', 'spiral-arm-count', 'spiral-winding', 'merging'],
                'problem': ['smooth-or-featured', 'problem', 'artifact']
            }
            selected_questions = branches[branch]

            question_dict = {
                'smooth-or-featured': {'smooth': 'merging', 'featured-or-disk': 'disk-edge-on', 'problem': 'problem'},
                'problem': {'star': None, 'artifact': None, 'zoom': None},
                'artifact': {'scattered': None, 'diffraction': None, 'ray': None, 'saturation': None, 'other': None, 'ghost': None},

                'how-rounded': {'round': 'edge-on-bulge', 'in-between': 'edge-on-bulge', 'cigar-shaped': 'edge-on-bulge'},
                'edge-on-bulge': {'boxy': 'merging', 'none': 'merging', 'rounded': 'merging'},

                'disk-edge-on': {'yes': 'merging', 'no': 'bar'},
                'bar': {'strong': 'spiral-arm-count', 'weak': 'spiral-arm-count', 'no': 'spiral-arm-count'},
                'has-spiral-arms': {'yes': 'spiral-arm-count', 'no': 'spiral-arm-count'},
                'spiral-arm-count': {'1': 'spiral-winding', '2': 'spiral-winding', '3': 'spiral-winding', '4': 'spiral-winding'},
                'spiral-winding': {'tight': 'merging', 'medium': 'merging', 'loose': 'merging'},
                'bulge-size': {'none': 'merging', 'small': 'merging', 'moderate': 'merging', 'large': 'merging', 'dominant': 'merging'},

                'merging': {'merger': None, 'major-disturbance': None, 'minor-disturbance': None, 'none': None}
            }
            question_dict = {k: v for k, v in question_dict.items() if k in selected_questions}

            question_values = deepcopy(question_dict)
            for q, answers in question_dict.items():
                for a in answers.keys():
                    question_values[q][a] = np.array([0, 1])
            logging.warning(question_dict)
            logging.info(question_values)
            
            self.question_dict = question_dict
            self.question_values = question_values

        def update_slider(self, question, answer, value):
            self.question_values[question][answer] = value




        


# questions, each with answers
# answers, each with value and next question



def interactive_galaxies(df):

    st.sidebar.markdown('# Choose Your Galaxies')

    interest = st.sidebar.radio("I'm interested in...", ['Featured galaxies', 'Smooth galaxies', 'Everything else'], key='interest')
    branch_renamer = {
        'Featured galaxies': 'featured',
        'Smooth galaxies': 'smooth',
        'Everything else': 'problem'
    }
    branch = branch_renamer[interest]

    questions = Questions(branch)

    # work out which to show by always showing first question
    # display_question(questions, 'smooth-or-featured')
    #     smooth_featured_answers = questions.question_values['smooth-or-featured']
    #     if smooth_featured_answers['smooth'] > 0.5:
    #         branch 

        
    for question in questions.question_dict.keys():
        display_question(questions, question)

    logging.info(questions.question_values)

    is_valid = find_valid_galaxies(df, questions)
    logging.info('Valid galaxies: {}'.format(is_valid.sum()))

    # with col1:
    # extra_filters = st.toggle('Extra filters', value=False)
    col1, _, _, _ = st.columns(4)
    with col1:
        view_preference = st.selectbox('View mode', ['Prioritise Large', 'Custom Filters'])

    if view_preference == 'Custom Filters':
        col1, col2, _, _ = st.columns(4)
        with col1:
                # actually this is area, roughly, via A = pi r **2 
                # min_radius = st.number_input('Minimum radius (arcsec)', min_value=7., value=30.)
                # min_area = np.pi * min_radius ** 2 # approx
            min_area = st.number_input('Minimum galaxy area (pixels, 0.1"/px)', min_value=700., value=3000.)
        with col2:
            faintest_mag = st.number_input('Faintest VIS mag', min_value=12., max_value=30., value=20.5)

        above_min_area = df['segmentation_area'] > min_area
        bright_enough = df['mag_segmentation'] < faintest_mag
        # bright_enough = True
        is_valid = is_valid & above_min_area & bright_enough
        selected = df[is_valid].sample(np.min([is_valid.sum(), 16]))

    elif view_preference == 'Prioritise Large':
        selected = df[is_valid].nlargest(40, 'cutout_width_arcsec')
    else:
        raise ValueError(view_preference)

    st.markdown('{:,} of {:,} galaxies match your criteria.'.format(is_valid.sum(), len(is_valid)))

    display_galaxies_via_html(selected)

    # display_galaxies_via_streamlit(selected)

def display_galaxies_via_streamlit(df, ncols=4, width=200):
    df['url'] = get_urls(df)
    urls = list(df['url'])
    captions = list(df['id_str'])
    # st.image(urls, caption=captions, width=200)
    # opening_html = '<div style=display:flex;flex-wrap:wrap>'
    # closing_html = '</div>'

    # st.html(opening_html)

    st.markdown("""
        <style>
        h1 {
            font-size: 16px;
            text-align: center;
            text-transform: uppercase;
        }
        </style>
        """, unsafe_allow_html=True)

    for i, row in df.iterrows():
        st.image(row['url'], caption=row['id_str'], width=200)

    # st.html(closing_html)

    # columns = st.columns(ncols)
    # for i, col in enumerate(columns):
    #     with col:
    #         st.image(urls[i], caption=captions[i], use_container_width=True)

def display_galaxies_via_html(df):
    # https://storage.googleapis.com/zootasks_test_us/euclid/q1_v5/cutouts_jpg_gz_arcsinh_vis_y/102018667/102018667_NEG585598010511040774_gz_arcsinh_vis_y.jpg
    df['url'] = get_urls(df)

    st.markdown("""
        <style>
        img:hover {
            border: 5px solid #FF4B4B;
            border-radius: 5px;
        }
        img {
                margin: 3px;
                width: 200px;
        }
        .galaxy {
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                align-items: center;
                margin: 0 auto;
                padding: 0;
                list-style: none;
        }
        input {
            background-color: #FF4B4B;
            border: none;
            color: white;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    # opening_html = '<div style=display:flex;flex-wrap:wrap>'
    # closing_html = '</div>'
    opening_html = '<div class="galaxy">'
    closing_html = '</div>'
    # child_html = [f'<a href="{url}"><img src="{url}" style=margin:3px;width:200px;></img></a>' for url in image_urls]
    child_html = []
    for _, row in df.iterrows():
        child_html += [
            f'''
                <ul>
                    <a href={row['url']}><img src="{row['url']}";></img></a>
                    <div>
                    <center>
                        <a href={get_legacy_url(row['right_ascension'], row['declination'])}>
                            <input type="submit" value=" {row['right_ascension']:.4f}, {row['declination']:.4f} "></input>
                        </a>
                        </center>
                    </div>
                </ul>'''
        ]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    # st.markdown(gallery_html)
    st.markdown(gallery_html, unsafe_allow_html=True)

def get_urls(df):
    df['tile_index'] = df['id_str'].apply(lambda x: x.split('_')[2])
    df['adjusted_id_str'] = df['id_str'].apply(lambda x: x.replace('-', 'NEG').replace('Q1_R1_', ''))
    image_urls = f'https://storage.googleapis.com/zootasks_test_us/euclid/q1_v5/cutouts_jpg_gz_arcsinh_vis_y/' + df['tile_index'] + '/' + df['adjusted_id_str'] + '_gz_arcsinh_vis_y.jpg'
    return image_urls

def find_valid_galaxies(df, questions):

    logging.debug('Total galaxies: {}'.format(len(df)))
    valid = np.ones(len(df)).astype(bool)  # all df are valid to start with, we will require AND for each filter

    for question, answer_limit_pairs in questions.question_values.items():
        # logging.debug(f'Current: {question}, {answer_limit_pairs}')
 
        # logging.debug('answer limit pairs: {}'.format(answer_limit_pairs))
        for answer, limit_pairs in answer_limit_pairs.items():
            logging.debug('{} limit pairs: {}'.format(question, limit_pairs))
            answer_col = question + '_' + answer + '_fraction'
            this_answer = df[answer_col]
            slider_not_used = limit_pairs[0] == 0 and limit_pairs[1] == 1
            within_limits = (limit_pairs[0] <= this_answer) & (this_answer <= limit_pairs[1])

            valid = valid & (slider_not_used or within_limits)

    logging.debug('Valid galaxies: {}'.format(valid.sum()))
    return valid

def display_question(questions, question):

    answers = list(questions.question_dict[question].keys())
    logging.warning('answers: {}'.format(answers))
    # will be displayed, even though not explicitly used
    selected_question = st.sidebar.markdown('## ' + question.replace('-', ' ').replace('_', ' ').capitalize() + '?')
    # returns the selected option
    selected_answer = st.sidebar.selectbox('Answer', answers, format_func=lambda x: x.replace('-',' ').capitalize(), key=question+'_select')
    # we don't need the key, only the limits
    selected_limits = st.sidebar.slider(
                label='Vote Fraction',
                min_value=0.,
                max_value=1.,
                value=(.0, 1.),
                key=question+'_slider')
    
    logging.debug('before edit: {}'.format(selected_limits))
    assert not isinstance(selected_limits, float)

    # if isinstance(selected_limits, float):
    # #     # streamlit is giving only the higher value since only higher end modified
    #     logging.warning('{} {}'.format(question, selected_limits))
    #     selected_limits = np.array((0., selected_limits))
    #     logging.debug('after edit: {}'.format(selected_limits))

    questions.question_values[question][selected_answer] = selected_limits


st.set_page_config(
    layout="wide",
    page_title='GZ Euclid',
    page_icon='gz_icon.jpeg'
)

@st.cache_data
def load_data():
    df = pd.read_parquet('morphology_catalogue_minimal.parquet')

    # df = df.dropna(subset=['has-spiral-arms_yes_fraction'])  # temp, only look at featured galaxies (new slider needed?)
    # max out cutout size, don't want to blow up bandwidth
    df['cutout_width_arcsec'] = df['cutout_width_arcsec'].clip(upper=130)  # 112 = 90th pc, 147 = 95th
    return df

def get_field(declination):
    if declination > 40:
        return 'EDFN'
    elif declination > -20:
        return 'TODO'  # not on esa sky
    elif declination < -40:
        return 'EDFF'
    else:
        return 'EDFS'

def get_esasky_url(ra, dec):
    logging.info(f'ra: {ra}, dec: {dec}')
    # hips_name = 'Q1-EDFS-R4-PNG-RGB'
    image_name = get_field(dec)
    # esasky example
# https://sky.esa.int/esasky/?target=61.091466763768565%20-47.95083211877482&hips=Q1-EDFS-R4-PNG-RGB&fov=0.24313710346433015&projection=TAN&cooframe=J2000&sci=false&lang=en&euclid_image=EDFS

    url = f'https://sky.esa.int/esasky/?target={ra}%20{dec}&hips=Q1-{image_name.upper()}-R4-PNG-RGB&fov=0.01&projection=TAN&cooframe=J2000&sci=false&lang=en&euclid_image={image_name}'
    logging.info(url)
    return url

def get_legacy_url(ra, dec):
    # https://www.legacysurvey.org/viewer?ra=168.0520&dec=27.6248&layer=ls-dr10&zoom=16
    url = f'https://www.legacysurvey.org/viewer?ra={ra}&dec={dec}&layer=ls-dr10&zoom=16'
    return url

if __name__ == '__main__':

    # logging.basicConfig(level=logging.CRITICAL)
    logging.basicConfig(level=logging.INFO)



    questions = Questions('featured')

    df = load_data()
    main(df)

    # import streamlit as st

    # values = st.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
    # st.write("Values:", values)


# https://discuss.streamlit.io/t/values-slider/9434 streamlit sharing has a temp bug that sliders only show the top value