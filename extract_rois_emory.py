# Purpose: Copy images from the HITI dataset to a new directory, with the ROI selected out.
# This extracts ROI for all of the images necessary for both test and tuning the context window size.
# EmoryDataset.ipynb symlinks the images into the directories that match IAIA-BL's expected directory configuration.
# (this was done to deal with IO limitations at the time)

import pandas as pd
import numpy as np
import re
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as patches
import os
# import imageio
# import aiofiles
import asyncio
import re
# import cv2
import logging
import urllib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from IPython.display import display
def display(*args):
    pass

from ast import literal_eval
from pathlib import Path
import time
import pickle
import PIL

import shutil
import sys

from random import sample

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

tqdm.pandas()
basepath = '/redacted/path/to/tables'

logger = logging.getLogger(__name__)

device = torch.device(f'cuda:{6}') if torch.cuda.is_available() else torch.device('cpu')

def stats(df, patient_column=None):

    if patient_column is not None:
        patient_column = df[patient_column]
    elif 'empi_anon' in df.index.names:
        patient_column = df.index.to_frame()["empi_anon"]
    else:
        patient_column = df["empi_anon"]

    stats_dict = {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Unique Patients": patient_column.nunique(),
    }
    
    if 'acc_anon_scr' in df.columns:
        stats_dict["Unique Screening Exams"] = df["acc_anon_scr"].nunique()
    if 'acc_anon_diag' in df.columns:
        stats_dict["Unique Diagnostic Exams"] = df["acc_anon_diag"].nunique()
    if 'acc_anon' in df.columns:
       stats_dict["Unique Exams"] = df["acc_anon"].nunique()

    return pd.DataFrame(stats_dict, index=['counts'])

def lead_with_columns(df, lead_columns):
    """
    Rearranges dataframe column order by pulling a few columns to the front but keeping the order for other columns
    """
    post_columns = [col for col in df.columns if col not in lead_columns]
    return df[lead_columns + post_columns]

def col_search(df, pattern, ignore_case=True):
    """
    Find columns that match a pattern
    """
    flag = re.NOFLAG if ignore_case == False else re.I
    return [c for c in df.columns if re.match(pattern, c, flags = flag)]

def metadata_for(df_legend, cols):
    """
    Generate a dataframe with the metadata for a particular column in the embed dataset.
    """
    return df_legend[df_legend['Header in export']
          .isin(cols)] \
          .sort_values('Header in export')

def basic_count(df, cols):
    if 'acc_anon' in df.columns:
        return df[cols + ['empi_anon',  'acc_anon']].groupby(cols, dropna=False).count()
    

KEY_COLUMNS_MAG = ['index', 'empi_anon', 'acc_anon', 'side', 'numfind', 'desc', 'asses']
KEY_COLUMNS_MET = ['Unnamed: 0', 'empi_anon', 'acc_anon', 'ImageLateralityFinal', 'ViewPosition', 'FinalImageType', 'AcquisitionTime']
CONTEXT_COLUMNS = ['cohort_num', 'study_date_anon']

def load_met(file_path=os.path.join(basepath, 'metadata_all_cohort_with_ROI_HITI.csv'), 
             scope='min', extra_cols=[], cohorts=None):
    
    force_string_cols = ['CollimatorShape', '0_ProcedureCodeSequence_CodeValue', 'DerivationDescription',
                         'CommentsOnRadiationDose', 'DetectorDescription', 'WindowCenter', 'WindowWidth']
    
    categorical_cols = ['AcquisitionDeviceProcessingCode', 'AcquisitionDeviceProcessingDescription',
                        'DetectorConfiguration', 'FieldOfViewShape', 'CollimatorShape',
                        'DetectorActiveShape', 'ExposureStatus', 'VOILUTFunction'
                        '0_IconImageSequence_PhotometricInterpretation']
    
    tuple_cols = ['FieldOfViewDimensions', 'DetectorActiveDimensions', 'DetectorElementPhysicalSize',
                  'DetectorElementSpacing', 'WindowCenterWidthExplanation']
    
    date_cols = ['study_date_anon']
    
    forced_types = {col: 'string' for col in force_string_cols}
    forced_types.update({col: 'category' for col in categorical_cols})
    forced_types.update({col: 'string' for col in tuple_cols})
    
    if scope == 'min':
        usecols = (KEY_COLUMNS_MET + CONTEXT_COLUMNS + 
                    ['png_path', 'png_filename', 'num_roi', 'ROI_coords', 'match_level', 'spot_mag'] + extra_cols)
    
    elif scope == 'full':
        usecols = None
    
    else:
        return NotImplemented(f'{scope} is not a valid column scope')
    
    df_metadata = pd.read_csv(file_path, dtype=forced_types, parse_dates=date_cols, usecols=usecols)
    
    if cohorts:
        df_metadata = df_metadata.loc[df_metadata['cohort_num'].isin(cohorts)]

    return df_metadata.pipe(lead_with_columns, KEY_COLUMNS_MET + CONTEXT_COLUMNS)

__birads_map = {
    'A': 0,
    'N': 1,
    'B': 2,   
    'P': 3,
    'S': 4,
    'M': 5,
    'K': 6,
    'X': 'X'
}
    # BIRADS 0: A – Additional evaluation
    # BIRADS 1: N – Negative
    # BIRADS 2: B - Benign
    # BIRADS 3: P – Probably benign
    # BIRADS 4: S – Suspicious
    # BIRADS 5: M- Highly suggestive of malignancy
    # BIRADS 6: K - Known biopsy proven

def translate_birads(hiti_asses):
    return __birads_map[hiti_asses]


def load_mag(file_path=os.path.join(basepath, 'magview_all_cohorts_anon_HITI.csv'), 
             scope = 'min', extra_cols=[], cohorts=None):
    # columns that have inconsistent value types - default casting doesn't work properly
    inconsistent_columns = ['case', 'biopsite', 'bcomp', 'path7', 'path8', 'path9', 'path10', 'hgrade',
                            'tnmpt', 'tnmpn', 'tnmm', 'tnmdesc', 'stage', 'bdepth','focality',
                            'specinteg', 'specembed', 'her2', 'fish', 'extracap', 'methodevl',
                            'eic', 'first_3_zip']
    type_overrides = {col: 'string' for col in inconsistent_columns}

    categories = ['massshape', 'massmargin', 'massdens', 'calcfind', 'calcdistri', 'otherfind', 
                  'implanfind', 'side', 'location', 'depth', 'distance', 'asses', 'recc', 'proccode',
                  'vtype','tissueden', 'MARITAL_STATUS_DESC']
    type_overrides.update({col: 'category' for col in categories})

    date_cols = {'study_date_anon','sdate_anon','procdate_anon','pdate_anon'}
    
    if scope == 'min':
        usecols = (KEY_COLUMNS_MAG + CONTEXT_COLUMNS + 
            ['massshape', 'massmargin', 'otherfind', 'path_group', 'path_severity'] + extra_cols)
        date_cols = date_cols.intersection(usecols)
        
    elif scope == 'full':
        usecols = None
        
    else:
        return NotImplemented(f'{scope} is not a valid column scope')
    
    df_mag =  pd.read_csv(file_path, dtype=type_overrides, header=0,
                          parse_dates=list(date_cols), usecols=usecols)
    
    df_mag['scr_or_diag'] = df_mag['desc'].apply(lambda desc: 'S' if 'screen' in desc.lower() else 'D').astype("category")
    df_mag['birads'] = df_mag['asses'].apply(translate_birads)

    if cohorts:
        df_mag = df_mag.loc[df_mag['cohort_num'].isin(cohorts)]

    return df_mag.pipe(lead_with_columns, KEY_COLUMNS_MAG + CONTEXT_COLUMNS)


def select__goldscr(def_met, image_type='2D'):
    """
    Selects the 'gold' screening images for exams according to our selection criteria so that each of the 4 views only has one image.
    """
    def_met_goldscreen = def_met[(def_met['ViewPosition'].isin(['MLO', 'CC'])) 
                              & (def_met['ImageLateralityFinal'].isin(['L', 'R']))
                              & (def_met['FinalImageType'] == image_type)
                              & (def_met['spot_mag'].isna())
                              & (def_met['scr_or_diag'] == 'S')]
    
    if def_met.index.name == 'acc_anon':
        def_met['acc_anon'] = def_met.index

    def_met_goldscreen = def_met_goldscreen.sort_values(['AcquisitionTime'], ascending=False)
    def_met_goldscreen = def_met_goldscreen.drop_duplicates(['ViewPosition', 'ImageLateralityFinal'])

    return def_met_goldscreen

# Initial the Datasets
df_mag = load_mag()
display(df_mag.head(4))

df_met = load_met(extra_cols=['ImageType',
'FilterType',
'WindowCenter',
'WindowWidth',])
display(df_met.head(4))

# Screen and Diagnostic Subsets
df_mag_scr = df_mag.loc[df_mag['scr_or_diag'] == 'S']
display(df_mag_scr.pipe(stats))

df_mag_diag = df_mag.loc[df_mag['scr_or_diag'] == 'D']
display(df_mag_diag.pipe(stats))


# Number of Screening Exams
scr_exam_ids = set(df_mag_scr['acc_anon'])
df_met_scr = df_met.loc[df_met.acc_anon.isin(scr_exam_ids)]
display(len(scr_exam_ids))
display(df_met_scr.pipe(stats))
# notice the loss of 14k 'screening' exams that have no images

# Get Exams with ROIs
df_met_scr_roi = df_met_scr.loc[df_met_scr.num_roi > 0]
display(df_met_scr_roi.pipe(stats))
scr_exam_wroi_ids = set(df_met_scr_roi.acc_anon)
len(scr_exam_wroi_ids)

# This count represents the number of ROIs in an image for all images with ROIs
# it is NOT normalized to the exam
df_met_scr_roi.pipe(basic_count, ['num_roi'])

df_met_scr_roi_2d = df_met_scr_roi.loc[df_met_scr_roi['FinalImageType'] == '2D']
df_met_scr_roi_2d.pipe(stats)

# Expand the dataset so that side='B' is treated as L and R
df_mag_scr_sideB = df_mag_scr.loc[df_mag_scr.side == 'B']
df_mag_scr_sideLR = df_mag_scr.loc[df_mag_scr.side != 'B']

display('LR Only', df_mag_scr_sideLR.pipe(stats))

for side in ['L', 'R']:
    df_mag_scr_side = df_mag_scr_sideB.copy()
    df_mag_scr_side['side'] = side
    display('with ' + side, df_mag_scr_side.pipe(stats))
    df_mag_scr_sideLR = pd.concat([df_mag_scr_sideLR, df_mag_scr_side])

df_mag_scr_B = df_mag_scr_sideLR
display('All', df_mag_scr_B.pipe(stats))

df_magXmet_scr_Wroi_2d = df_met_scr_roi_2d.merge(df_mag_scr_B,
                                    left_on=['acc_anon', 'empi_anon', 'ImageLateralityFinal'], 
                                    right_on=['acc_anon', 'empi_anon', 'side'], 
                                    suffixes=('_met', '_mag'))
df_magXmet_scr_Wroi_2d['study_date_anon'] = df_magXmet_scr_Wroi_2d['study_date_anon_mag']

# This is about 300 less exams than have ROIs
display(df_magXmet_scr_Wroi_2d.pipe(stats))
df_magXmet_scr_Wroi_2d.pipe(basic_count, ['asses', 'num_roi', 'side'])

df_magXmet_scrXdiag_Wroi_2d = df_magXmet_scr_Wroi_2d.merge(df_mag_diag, on=['empi_anon', 'side'], suffixes=['_scr', '_diag'])
display(df_magXmet_scrXdiag_Wroi_2d.columns, df_magXmet_scrXdiag_Wroi_2d.pipe(stats))

# This generates a filtering function so you can select a different month ranges to compare, if you like
def filter_forward_date_for(source_col, filter_col, months_range=6):
    def filter_by_date_range(row):
        start_date = row[source_col] 
        end_date = row[source_col] + pd.DateOffset(months=months_range)
        return (row[filter_col] >= start_date) & (row[filter_col] <= end_date)
    return filter_by_date_range

for months in [6]:
    proc_date_filter = filter_forward_date_for('study_date_anon_scr','study_date_anon_diag', months_range=months)
    df_magXmet_scrXdiag_Wroi_2d[f'screen_{months}_before_diag'] = df_magXmet_scrXdiag_Wroi_2d.apply(proc_date_filter, axis=1)
    
df_magXmet_scrXdiag_WroiW6monfollow_2d = df_magXmet_scrXdiag_Wroi_2d[df_magXmet_scrXdiag_Wroi_2d['screen_6_before_diag']].drop_duplicates() \
    .pipe(lead_with_columns, ['empi_anon', 'acc_anon_scr', 'study_date_anon_scr', 'acc_anon_diag', 'study_date_anon_diag', 'path_severity_scr', 'path_severity_diag'])

display(df_magXmet_scrXdiag_WroiW6monfollow_2d.pipe(stats))
df_magXmet_scrXdiag_WroiW6monfollow_2d.head(3)


def gold_margin(row):
    if type(row['massmargin_diag']) != float:
        return row['massmargin_diag']
    else:
        return row['massmargin_scr']

df_magXmet_scrXdiag_WroiW6monfollow_2d['worst_path_severity'] = df_magXmet_scrXdiag_WroiW6monfollow_2d[['path_severity_scr', 'path_severity_diag']].min(axis=1)
df_magXmet_scrXdiag_WroiW6monfollow_2d['massmargin_gold'] = df_magXmet_scrXdiag_WroiW6monfollow_2d.apply(gold_margin, axis=1)
df_magXmet_scrXdiag_WroiW6monfollow_2d.groupby(['worst_path_severity', 'massmargin_gold']).count()[['empi_anon']]

df_magXmet_scrXdiag_WroiW6monfollow_2d = df_magXmet_scrXdiag_WroiW6monfollow_2d.drop_duplicates(['png_path'])

iaiabl_margins = {
    'D': 'Circumscribed',
    'I': 'Indistinct',
    'S': 'Spiculated'
}


df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d =\
    df_magXmet_scrXdiag_WroiW6monfollow_2d.loc[df_magXmet_scrXdiag_WroiW6monfollow_2d['massmargin_gold'].isin(iaiabl_margins.keys())]

df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d.groupby(['massmargin_gold', 'worst_path_severity']).count()[['empi_anon']]


from collections import namedtuple
from ast import literal_eval as make_tuple

roi = namedtuple('roi', ['top', 'bottom', 'dist'])

def parse_roi(row):

    pre_tuples = make_tuple(row['ROI_coords'])
    if len(pre_tuples) == 0:
        return []
    
    rois = []
    tuples = []
    for maybe_tuple in pre_tuples:
        if len(maybe_tuple) == 0:
            print('empty tuple:', maybe_tuple)
            continue

        if type(maybe_tuple[0]) == list:
            for tup in maybe_tuple:
                tuples.append(tup)
        else:
            tuples.append(maybe_tuple)
    
    for t in tuples:
        
        try:
            y_dist = t[2] - t[0]
            x_dist = t[3] - t[1]

            bottom = (t[3], t[2])
            top= (t[1], t[0])

            rois.append(roi(top, bottom, (x_dist, y_dist)))
        except Exception as e:
            print(row, t)
            print(row['ROI_coords'])
            raise e
        
    return rois

df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d['roi_tuples'] = df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d.apply(parse_roi, axis=1)

df_magXmet_scrXdiag_WroiW6monfollowWiaiablmarginWwindow_2d = df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d.loc[df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d['WindowCenter'] == '2047.0']

# Average Window Centers, Maximum Window Width
df_magXmet_scrXdiag_WroiW6monfollowWiaiablmarginWwindow_2d = df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d.loc[df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d['WindowCenter'] != '2047.0']

df_magXmet_scrXdiag_WroiW6monfollowWiaiablmarginWwindow_2d.groupby(['WindowCenter', 'WindowWidth']).count()[['empi_anon']].sort_values('empi_anon', ascending=False)


# from preprocess import mean, std
# import torchvision.transforms as transforms
import png
import imageio

# normalize = transforms.Normalize(mean=np.array(mean).mean(), std=np.array(std).mean())
# print(normalize)
# import torchvision.transforms.functional as F

def read_16bit_img(path):
    # read image into np
    reader = png.Reader(path)
    data = reader.read()
    print(data)
    pixels = data[2]
    image = []
    for row in pixels:
        row = np.asarray(row, dtype=np.float32)
        image.append(row)
    return np.stack(image, 1)

def windowing(wcen, wwidth, img):
    img = (img - img.min()) / img.max()
    return np.clip(img,0,1)
    # print('wcen', wcen, 'wwdith', wwidth, 'img', img.shape)
    # recentered = (img - wcen)
    # print('recenter', recentered.max(), recentered.min(), recentered.mean())
    # normed = (recentered / float(wwidth))
    # print('normed', normed.max(), normed.min(), normed.mean())
    # image = normed + 0.5
    # return np.clip(image, 0, 1)

def load_img(png_path):
    return np.asarray(PIL.Image.open(png_path).convert('F'))
    # return np.asarray(imageio.imread(png_path, format='png', as_gray=True)).astype(np.float32)
    # return read_16bit_img(png_path)

def save_img(save_path, img):
    np.save(save_path, img)
    return Path(save_path)

def stupid_normalize(im):
    im_arr = np.asarray(im.convert('F')).astype(np.float64)

    mean_ratio =  im_arr.mean()/np.array(mean).mean()
    im_arr = im_arr/mean_ratio
    std_dev_ratio = im_arr.std()/np.array(std).mean() * 1.4

    im_arr = im_arr - (im_arr - im_arr.mean())/std_dev_ratio

    if im_arr.min() <= 0.0:
        im_arr = im_arr + im_arr.min()

    if im_arr.max() >= 1.0:
        im_arr = im_arr/im_arr.max()

    return im_arr


def roi_context_window(image, roi, do_torch=False):

    x_top, y_top = roi.top
    x_bot, y_bot = roi.bottom
    height, width = image.shape

    ctx = 100

    x_ctx_top = max(x_top-ctx, 0)
    y_ctx_top = max(y_top-ctx, 0)

    x_rel_top = x_top - x_ctx_top
    y_rel_top = y_top - y_ctx_top

    x_ctx_bot = min(x_bot+ctx, width)
    y_ctx_bot = min(y_bot+ctx, height)

    x_rel_bot = x_rel_top + roi.dist[0]
    y_rel_bot = y_rel_top + roi.dist[1]

    if do_torch:
        roi_box = torch.ones((x_ctx_bot - x_ctx_top, y_ctx_bot-y_ctx_top), dtype=torch.float64).cuda()
        roi_box[x_rel_top:x_rel_bot, y_rel_top:y_rel_bot] = 0

        box = (x_ctx_top, y_ctx_top, x_ctx_bot, y_ctx_bot)
        im_arr = torch.tensor(image, dtype=torch.float64).cuda()[y_ctx_top:y_ctx_bot, x_ctx_top:x_ctx_bot]

        print(roi_box.shape, im_arr.shape)

        return torch.cat((torch.unsqueeze(im_arr, dim=0), torch.unsqueeze(roi_box.T, dim=0)), dim=0).cpu().numpy()

    else:
        roi_box = np.ones((x_ctx_bot - x_ctx_top, y_ctx_bot-y_ctx_top)).astype(np.float64)
        roi_box[x_rel_top:x_rel_bot, y_rel_top:y_rel_bot] = 0

        box = (x_ctx_top, y_ctx_top, x_ctx_bot, y_ctx_bot)
        im_arr = image[y_ctx_top:y_ctx_bot, x_ctx_top:x_ctx_bot]

        print(roi_box.shape, im_arr.shape)

        return np.concatenate((np.expand_dims(im_arr, axis=0), np.expand_dims(roi_box.T, axis=0)), axis=0)

row = df_magXmet_scrXdiag_WroiW6monfollowWiaiablmarginWwindow_2d.iloc[27]
img = load_img(row['png_path'])

plt.figure()
f, axarr = plt.subplots(1,4) 
axarr[0].imshow(img, cmap='gray')

window_center, window_width = None, None #float(row.WindowCenter), float(row.WindowWidth)
print(window_center, window_width, img.max(), img.min())
img_w = windowing(window_center, window_width, img)
print(img_w.max(), img_w.min())

axarr[1].imshow(img_w, cmap='gray')
np_img = roi_context_window(img_w, row['roi_tuples'][0], do_torch=False)
np_img.shape

print(np_img[0].max(), np_img[0].min(), np_img[1].max(), np_img[1].min())
np_img

print(np_img.shape)

np_img[-1,-1] = 1
np_img[-1,-2] = 0

axarr[2].imshow(np_img[1], cmap='gray')
axarr[3].imshow(np_img[0], cmap='gray')

print(np_img[0].min(), np_img[0].max(), np_img[0].mean(), np_img[0].shape, np_img[0].dtype)

plt.xlabel(f'Mask for ')
plt.show()


from tqdm import tqdm
def copy_rois(grouped_df, target_dir):

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)  # create the target directory if it doesn't exist

    final_paths = {}

    for group_name, group in grouped_df:
        margin = iaiabl_margins[group_name]
        group_path = target_path / margin
        group_path.mkdir(parents=True, exist_ok=True)
        for row in tqdm(group.itertuples()):
            try:
                img = load_img(row.png_path)
                try:
                    window_center, window_width = None, None #float(row.WindowCenter), float(row.WindowWidth)
                except:
                    print(f'non singular window for row {row}')
                    continue

                img = windowing(window_center, window_width, img)

                final_paths[row.empi_anon] = []
                for i, roi in enumerate(row.roi_tuples):
                    file_name = Path(row.png_path).name + f"_{i}"

                    roi_img = roi_context_window(img, roi, do_torch=True)
                    final_paths[row.empi_anon].append( save_img(group_path / file_name, roi_img).with_suffix('.npy') )
            except Exception as e:
                print(f'failed to preprocess row {row} with exception {e}')

            
    return final_paths

print("*******")
print("*******")
print("*******")
print("*******")
print("*******")
print("*******")
print("Starting Copy!")
paths = copy_rois(df_magXmet_scrXdiag_WroiW6monfollowWiaiablmargin_2d.groupby(['massmargin_gold']), '/redacted/path/to/output')