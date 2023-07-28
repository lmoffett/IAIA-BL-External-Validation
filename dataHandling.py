from __future__ import division
import numpy as np
import os
import pandas as pd
import argparse
import sys
import random
import png
from matplotlib.pyplot import imsave, imread
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2
matplotlib.use("Agg")
import torchvision.datasets as datasets
from skimage.transform import resize
import ast
import pickle
import csv
import pydicom as dcm
import Augmentor
from tqdm import tqdm
import pathlib
from torch import randint, manual_seed
from copy import copy
from collections import defaultdict, namedtuple
from bs4 import BeautifulSoup
import re
import configparser
from ast import literal_eval
import csv
import pydicom
from matplotlib.patches import Rectangle

def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input

def random_crop(input, with_fa=False):
    ran = random.random()
    if ran > 0.2:
        # find a random place to be the left upper corner of the crop
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input

def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1,2))
        return np.rot90(input)
    else:
        return input

def random_rotation(x, chance, with_fa=False):
    ran = random.random()
    if with_fa:
        img = Image.fromarray(x[0])
        mask = Image.fromarray(x[1])
        if ran > 1 - chance:
            # create black edges
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
            return np.stack([np.asarray(img), np.asarray(mask)])
        else:
            return np.stack([np.asarray(img), np.asarray(mask)])
    img = Image.fromarray(x)
    if ran > 1 - chance:
        # create black edges
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
        return np.asarray(img)
    else:
        return np.asarray(img)

def augment_numpy_images(path, targetNumber, targetDir, skip=None, rot=True, with_fa=False):
    classes = os.listdir(path)
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    for class_ in classes:
        if not os.path.exists(targetDir + class_):
            os.makedirs(targetDir + class_)

    for class_ in classes:
        count, round = 0, 0
        while count < targetNumber:
            round += 1
            for root, dir, files in os.walk(os.path.join(path, class_)):
                for file in files:
                    if skip and skip in file:
                        continue
                    filepath = os.path.join(root, file)
                    arr = np.load(filepath)
                    print("loaded ", file)
                    print(arr.shape)
                    try:
                        arr = random_crop(arr, with_fa)
                        print(arr.shape)
                        if rot:
                            arr = random_rotation(arr, 0.9, with_fa)
                        print(arr.shape)
                        arr = random_flip(arr, 0, with_fa)
                        arr = random_flip(arr, 1, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        print(arr.shape)
                        if with_fa:
                            whites = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0] - np.amax(arr[0]), 2))
                            black = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0], 2))
                            if arr.shape[2] < 10 or arr.shape[1] < 10 or black >= arr.shape[2] * arr.shape[1] * 0.8 or \
                                whites >= arr.shape[2] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue

                        else:
                            whites = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 2))
                            black = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr, 2))

                            if arr.shape[0] < 10 or arr.shape[1] < 10 or black >= arr.shape[0] * arr.shape[1] * 0.8 or \
                                    whites >= arr.shape[0] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue

                        if count % 10 == 0:
                            if not os.path.exists("./visualizations_of_augmentation/" + class_ + "/"):
                                os.makedirs("./visualizations_of_augmentation/" + class_ + "/")
                            if with_fa:
                                imsave("./visualizations_of_augmentation/" + class_ + "/" + str(count), np.transpose(np.stack([arr[0], arr[0], arr[1]]), (1,2,0)))
                            else:
                                imsave("./visualizations_of_augmentation/" + class_ + "/" + str(count), np.transpose(np.stack([arr, arr, arr]), (1,2,0)))


                        np.save(targetDir + class_ + "/" + file[:-4] + "aug" + str(round), arr)
                        count += 1
                        print(count)
                    except:
                        print("something is wrong in try, details:", sys.exc_info()[2])
                        if not os.path.exists("./error_of_augmentation/" + class_ + "/"):
                            os.makedirs("./error_of_augmentation/" + class_ + "/")
                        np.save("./error_of_augmentation/" + class_ + "/" + str(count), arr)
                    if count > targetNumber:
                        break
    print(count)

def window_augmentation(wwidth, wcen):
    if wcen == 2047 and wwidth == 4096:
        return wwidth, wcen
    else:
        new_wcen = np.random.randint(-100, 300)
        new_wwidth = np.random.randint(-200, 300)
        wwidth += new_wwidth
        wcen += new_wcen
        return wwidth, wcen

#Clean
from pathlib import Path
def find_dicoms(directory_to_search):
    dcm_paths = []
    for root, dirs, files in os.walk(directory_to_search):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = Path(root) / file
                dcm_paths.append(dcm_path)
    return dcm_paths

def truthfile_for(dcm_path, base_path=Path('/redacted/truthfile')):
    return base_path / dcm_path.parts[5] / dcm_path.parts[6] / 'truth' / dcm_path.parts[8] / dcm_path.parts[9] / dcm_path.parts[10] / ('_'.join(dcm_path.parts[8:11]) + '.xml')

def icad_patient_from_path(path):
    path = path.resolve()

    if path.parts[5] == 'ann_txt' or path.parts[5] == 'dbt':
        assert len(path.parts) >= 7, 'patient ID is 8th subdirectory'
        return path.parts[8]
    elif path.parts[5] == 'aggregated_truth':
        assert len(path.parts) >= 8, 'patient ID is 9th subdirectory'
        return path.parts[9]
    else:
        raise Exception(f'Unknown pathspec for patient id {path}')

def icad_vendor_from_path(path):
    path = path.resolve()

    if path.parts[5] == 'ann_txt' or path.parts[5] == 'dbt':
        assert len(path.parts) >= 6, 'patient ID is 8th subdirectory'
        return path.parts[6]
    elif path.parts[5] == 'aggregated_truth':
        assert len(path.parts) >= 7, 'patient ID is 9th subdirectory'
        return path.parts[7]
    else:
        raise Exception(f'Unknown pathspec for patient id {path}')

def icad_side_view_from_path(path):
    if path.parts[5] == 'ann_txt' or path.parts[5] == 'dbt':
        assert len(path.parts) >= 6, 'patient ID is 8th subdirectory'
        spec = path.parts[10]
    elif path.parts[5] == 'aggregated_truth':
        assert len(path.parts) >= 7, 'patient ID is 9th subdirectory'
        spec = path.parts[11]
    else:
        raise Exception(f'Unknown pathspec for patient id {path}')

    #eg, RMLO_0
    trimmed = spec.split('_')[0]
    return trimmed[0], trimmed[1:]

def pandas_for_icad_vendor(dcm_root_path, limit=None):
    records = []
    for i, dcm_file in tqdm(enumerate(find_dicoms(dcm_root_path))):
        truthfile_path = truthfile_for(dcm_file)
        patient_id = icad_patient_from_path(dcm_file)
        vendor = icad_vendor_from_path(dcm_file)
        side, view = icad_side_view_from_path(dcm_file)
        record = {'dcm_file': dcm_file, 'truthfile': truthfile_path}
        if not truthfile_path.exists():
            record['truthfile_exists'] = False
            records.append(record)
        else:
            record['truthfile_exists'] = True
            with open(truthfile_path, 'r') as f:
                per_mass_records = parse_truth(f.read(), patient_id, vendor, side, view)
                for mass_record in per_mass_records:
                    this_record =record.copy()
                    this_record.update(mass_record)
                    records.append(this_record)
        
        if limit and i >= limit:
            break

    return pd.DataFrame(records)

def pandas_for_icad(path, vendors=['ge', 'hologic', 'siemens'], limit=None):
    vendor_dfs = []
    for vendor in vendors:
        print('loading data for', vendor)
        vendor_dfs.append(pandas_for_icad_vendor(path / vendor))

    return pd.concat(vendor_dfs)

def empty_el(xml_el):
    non_truthed_els = len(set(xml_el.attrs.keys()) - set('truthed'))
    return non_truthed_els > 0 or len(xml_el.text) > 0

def safe_text(el):
    if el is None:
        return None
    else:
        return None if el.text == '' else el.text    

def safe_attr(el, attr):
    if el is None:
        return None
    else:
        return el.attrs.get(attr, None)

def parse_truth(data, patient_id, vendor, side, view):
    xml_obj = o = BeautifulSoup(data, "xml")

    record_dict = r = {}

    r['patient_id'] = patient_id
    r['manufacturer'] = vendor
    r['side'] = side
    r['view'] = view
    #   <truthQC truthed="1" truther="CP" />
    r['truth_qc'] = safe_attr(o.truthQC, 'truthed')
    r['truth_qc_truther'] = safe_attr(o.truthQC, 'truther')
    r.update(__parse_data_verification(o.data_verification))
    #   <comments>test</comments>
    r['comments'] = safe_text(o.comments)
    #   <acquisition_type>1</acquisition_type>
    r['acquistion_type'] = safe_text(o.acquisition_type)


    # Existance
    #   <anatomical />
    r['has_anantomical'] = not empty_el(o.anatomical)
    #   <glandular truthed="1">2</glandular>
    r['glandular'] = o.glandular.text
    r['glandular_truthed'] = o.glandular.attrs.get('truthed', None)
    #   <birads truthed="1">4</birads>
    r['birads'] = o.birads.text
    r['birads_truthed'] = o.birads.attrs.get('truthed', None)
    #   <clusters truthed="0" />
    r['has_clusters'] = not empty_el(o.clusters)
    #   <mixed truthed="0" />
    r['has_mixed'] = not empty_el(o.mixed)
    #   <fps truthed="0" />
    r['has_fps'] = not empty_el(o.fps)

    annotation_path = Path('/redacted/basepath') / vendor / 'truth' / patient_id

    # Parse Downstreams
    record_dict.update(__parse_image_info(o.image_info))

    r['masses_truthed'] = o.masses.attrs.get('truthed', None)
    masses = o.masses.find_all('mass')
    r['masses_count'] = len(masses)

    if len(masses) == 0:
        records = [record_dict]
    else:
        records = []
        for mass in masses:
            this_record_dict = record_dict.copy()
            this_record_dict.update(__parse_mass(mass, annotation_path))
            records.append(this_record_dict)

    return records

def __parse_image_info(image_xml_obj):
    r = {}
    o = image_xml_obj
    #     <image_info>
    
    #     <modality>MG</modality>
    r['modality'] = safe_text(o.modality)
    #     <manufacturer>GE MEDICAL SYSTEMS</manufacturer>
    r['manufacturer'] = safe_text(o.manufacturer)
    #     <model_name>Senographe Essential ADS_56.21.3</model_name>
    r['model_name'] = safe_text(o.model_name)
    #     <height>3062</height>
    r['height'] = safe_text(o.height)
    #     <width>2394</width>
    r['width'] = safe_text(o.width)
    #     <depth>1</depth>
    r['depth'] = safe_text(o.depth)
    
    for v in ['x', 'y', 'z']:
        #     <pixel_spacing x="0.1" y="0.1" z="1.0" />
        r[f'pixel_spacing_{v}'] = safe_attr(o.pixel_spacing, v)
        #     <origin x="0.0" y="0.0" z="0.0" />
        r[f'origin_{v}'] = safe_attr(o.origin, v)
        #     <axis x0="1.0" x1="0.0" x2="0.0" y0="0.0" y1="1.0" y2="0.0" z0="0.0" z1="0.0" z2="1.0" />
        for i in range(3):
            r[f'axis_{v}{i}'] = safe_attr(o.axis, f'{v}{i}')
    #     </image_info>
    return {f'ii_{key}': value for key, value in r.items()}

def __parse_mass(mass_xml_obj, annotation_path):
    r = {}
    o = mass_xml_obj
    #   Existance
    
    #   <masses truthed="1">
    #     <mass detectable="0" lesion_uid="1" missed="0" uid="smitg0006604.1929916722461">
    r['detectable'] = o.attrs.get('detectable', None)
    r['lesion_uid'] = o.attrs.get('lesion_uid', None)
    r['missed'] = o.attrs.get('missed', None)
    r['uid'] = o.attrs.get('uid', None)
    #       <pathology>
    #         <type />
    #         <size />
    #       </pathology>

    r['pathology_type'] = safe_text(o.pathology.type)
    r['pathology_size'] = safe_text(o.pathology.size)
    #       <location>
    #         <quadrant />
    #         <clockface />
    #         <z_location />
    #       </location>
    r['quadrant'] = safe_text(o.location.quadrant)
    r['clockface'] = safe_text(o.location.clockface)
    r['z_location'] = safe_text(o.location.z_location)

    #       <annotation link="smitg0006604a_conventional_pres_LCC_0.1929916722461.ann" truthed="0" />
    anno_link = o.annotation.attrs.get('link', None)
    # r['anno_ind'] = safe_text(o.annotation) is not None

    if anno_link is not None:
        patient = anno_link.split('_')[0]
        r['anno_link'] = annotation_path / (anno_link + '.txt')
        r['anno_link_exists'] = r['anno_link'].exists()

        if r['anno_link_exists']:
            parse_annotation = __parse_path_annotation(r['anno_link'])
            r.update({f'anno_{k}':v for k,v in parse_annotation.items()})
        
    points =  o.annotation.find_all('point')
    r['path_len'] = len(points)
    if len(points) > 0:
        r['path_points'] = [point_3d(float(point.attrs['x']), float(point.attrs['y']), 0.0) for point in points]
    
    r['anno_truthed'] = o.annotation.attrs['truthed']
    
    #       <histo>1</histo>
    r['histo'] = safe_text(o.histo)

    #       <subtlety />
    r['subtlety'] = safe_text(o.subtlety)
    
    #       <conspicuity />
    r['conspicuity'] = safe_text(o.conspicuity)

    #       <size />
    r['size'] = safe_text(o.size)
    #       <birads>
    #         <score>4</score>
    #         <descs />
    #       </birads>
    r['birads_score'] = safe_text(o.birads.score)

    r['birads_desc'] = set()
    for desc in o.birads.descs.find_all('desc', recursive=False):
        r['birads_desc'].add(desc)

    #       <type>6002</type>
    r['type'] = [safe_text(t) for t in o.find_all('type', recursive=False)]

    #       <core />
    r.update(__parse_core(o.core, annotation_path))

    #       <spiculation>
    #         <annotation link="smitg0006604a_conventional_pres_LCC_0.1929916722461.ann" truthed="0" />
    #       </spiculation>
    r.update(__parse_spiculation(o.spiculation, annotation_path))

    #     </mass>
    #   </masses>
    return {f'mass_{key}': value for key, value in r.items()}

def __parse_mass_annotation(link, basepath):
    patient = link.split('_')[0]
    patient_path = basepath / (link + '.txt')

    if not patient_path.exists():
        r = {'annotation_file_exists': False}
    else:
        r = {'annotation_file_exists': True}

        with open(patient_path, 'r') as f:
            text = f.read()
            r['annotation_text'] = text
    
    return r

def __parse_data_verification(data_ver_xml_obj):
    #  <data_verification>
    #     <flag truthed="1" truther="DC">DDE</flag>
    #   </data_verification>

    if data_ver_xml_obj is None:
        return {}

    flags = set()
    for flag in data_ver_xml_obj.find_all('flag'):
        flags.add(flag.text)
    
    return {'data_verification_flags': flags}

point_3d = namedtuple('point_3d', ['x', 'y', 'z'])

def __parse_core(core_xml_obj, annotation_path):
    r = {}

    if core_xml_obj is not None and len(core_xml_obj.find_all('annotation')) > 0:
        r['core_anno_name'] = core_xml_obj.annotation.attrs.get('link', None)
        r['core_anno_link']  = (annotation_path / r['core_anno_name']).with_suffix('.ann.txt')
        r['core_anno_link_exists'] = r['core_anno_link'].exists()

        if r['core_anno_link_exists']:
            parse_annotation = __parse_path_annotation(r['core_anno_link'])
            r.update({f'core_anno_{k}':v for k,v in parse_annotation.items()})
        
        # Not needed - tested that it always matches the points with the exact values
        # Guessing this was done for backwards compatability
        # if core_xml_obj.annotation.polygon is not None:
            # r['core_poly_points'] = core_xml_obj.annotation.find('polygon').find_all('point')

        points = core_xml_obj.annotation.find_all('point')
        r['core_path_len'] = len(points)
        if len(points) > 0:
            r['core_path_points'] = [point_3d(float(point.attrs['x']), float(point.attrs['y']), 0.0) for point in points]

    return r

def __parse_spiculation(spiculation_xml_obj, annotation_path):
    r = {}

    if spiculation_xml_obj is not None and len(spiculation_xml_obj.find_all('annotation')) > 0:
        r['spiculation_anno_name'] = spiculation_xml_obj.annotation.attrs.get('link', None)

        r['spiculation_anno_link'] = (annotation_path / r['spiculation_anno_name']).with_suffix('.ann.txt')
        r['spiculation_anno_link_exists'] = r['spiculation_anno_link'].exists()

        if r['spiculation_anno_link_exists']:

            parse_annotation = __parse_path_annotation(r['spiculation_anno_link'])
            r.update({f'spiculation_anno_{k}':v for k,v in parse_annotation.items()})

        # Not needed - tested that it always matches the points with the exact values
        # Guessing this was done for backwards compatability
        # if spiculation_xml_obj.annotation.polygon is not None:    
        #     r['spiculation_poly_points'] = spiculation_xml_obj.annotation.find('polygon').find_all('point')

        points =  spiculation_xml_obj.annotation.find_all('point')
        r['spiculation_path_len'] = len(points)
        if len(points) > 0:
            r['spiculation_path_points'] = [point_3d(float(point.attrs['x']), float(point.attrs['y']), 0.0) for point in points]
                

    return r

def __parse_path_annotation(annotation_path):
    with open(annotation_path, 'r') as f:
        raw = f.read()
    
    parts = raw.split('\n')
    kw = '\n'.join(parts[:10])
        
    config = configparser.ConfigParser()
    config.read_string('[dummy]\n' + kw)
    r = {f'path_{k}':v for k,v in config['dummy'].items()}

    r['path_points'] = [point_3d(*literal_eval(tup.replace(' ', ','))) for tup in parts[10:] if len(tup) > 0]
    return r

def read_16bit_img(path):
    # read image into np
    reader = png.Reader(path)
    data = reader.read()
    pixels = data[2]
    image = []
    for row in pixels:
        row = np.asarray(row, dtype=np.float32)
        image.append(row)
    return np.stack(image, 0)

def windowing(wcen, wwidth, img):
    image = ((img - wcen) / float(wwidth)) + 0.5
    return np.clip(image, 0, 1)

def path_to_bounding_box(img_height, img_width, path_points):
    y1, x1, y2, x2 = int(img_height), int(img_width), 0, 0
    for p in path_points:
        y1 = min(y1, np.round(p.y))
        y2 = max(y2, np.round(p.y))
        x1 = min(x1, np.round(p.x))
        x2 = max(x2, np.round(p.x))

    return ((int(x1), int(y1)), (int(x2), int(y2)))

def extract_dicom_roi(dcm_path):
    ds = pydicom.dcmread(str(dcm_path))
    
    try:
        roi_x = ds.get(('0045', '1060'))
        roi_y = ds.get(('0045', '1061'))

        roi_x.value, roi_y.value
    except:
        print('no bb for', dcm_path)
        return None

    bb = [(int(x), int(y)) for x, y in zip(roi_x.value, roi_y.value)][::2]

    return bb

def extract_image_from_dicom(source_path, mass_bounding_box, target_path, make, path_points, context=100):
    ds = pydicom.dcmread(str(source_path))

    png_path = target_path.with_suffix('.png')
    numpy_path = target_path.with_suffix('.npy')
    plt_path = target_path.with_suffix('.plt.png')

    img = ds.pixel_array # get image array

    # # Get Windowing from DICOM
    def safe_eval(element_name):

        for ele in ds:
            if ele.name == element_name:
                try:
                    return int(ele.value)
                except:
                    try:
                        return [float(v) for v in ast.literal_eval(str(ele.value))]
                    except:
                        print('failed to extract', element_name, 'with value', ele.value)
                        raise
    
    win_cens = safe_eval("Window Center")
    win_widths = safe_eval("Window Width")

    # Write PNG so it can be read the same way as other environments
    cv2.imwrite(str(png_path), img) # write png image
    p_image = read_16bit_img(str(png_path)) # load png img

    if make=="ge":
        win_cen, win_width = sum(win_cens)/len(win_cens), max(win_widths)
        p_image = windowing(win_cen, win_width, p_image)
    elif make=="hologic" or make =="siemens":
        win_cen, win_width = win_cens, win_widths
        p_image = windowing(win_cens, win_widths, p_image)
    else:
        raise Exception(f'Manufacturer was "{make}"')

    plt.imsave(str(plt_path), p_image, cmap="gray", format="png")
    
    image = cv2.imread(str(plt_path))

    (x1, y1), (x2, y2) = mass_bounding_box
    # Bounding Box augmented with Context Window
    x1_, y1_, x2_, y2_ = ctx_bounding_box = (
        max(0, min(x1, x2) - context),
        max(0, min(y1, y2) - context),
        min(p_image.shape[1], max(x1, x2) + context),
        min(p_image.shape[0], max(y1, y2) + context)
    )
    
    roi = p_image[y1_:y2_, x1_:x2_]

    # FA Mask
    fa_mask = np.zeros(shape=p_image.shape)
    fa_mask[0: min(y1, y2), :] = 1
    fa_mask[:, 0: min(x1, x2)] = 1
    fa_mask[max(y2, y1):, :] = 1
    fa_mask[:, max(x2, x1):] = 1

    fa_mask = fa_mask[y1_:y2_, x1_:x2_]

    try:
        np.save(str(numpy_path), np.stack([roi, fa_mask])) #fa_mask
    except:
        print('stack failed', roi.shape, fa_mask.shape, p_image.shape, image.shape, mass_bounding_box, ctx_bounding_box)
        raise

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    for p in path_points:
        axs[0].plot(p.x, p.y, 'r.')

    bb = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec='green')
    axs[0].add_patch(bb)
    ctx = Rectangle((x1_, y1_), x2_-x1_, y2_-y1_, fill=False, ec='blue')
    axs[0].add_patch(ctx)

    axs[1].imshow(image[y1_:y2_, x1_:x2_])

    fig.savefig(plt_path)
    plt.close(fig)

    return {'win_cens': win_cens,
            'win_cen': win_cen,
            'win_widths': win_widths,
            'win_width': win_width,
            'png_path': png_path,
            'numpy_path': numpy_path,
            'plt_path': plt_path
        }

def make_icad_iaiabl_dataset_folder(image_metadata, target_path):

    for dirpath in [target_path, target_path/ "not_spiculated", target_path/ "spiculated"]:
        dirpath.mkdir(exist_ok=True)

    records = []

    for idx, dicom_path, manufacturer, margin, mass_uid, mass_bounding_box, path_points in tqdm(image_metadata[['dcm_file', 'manufacturer', 'mass_margin', 'mass_uid', 'path_bounding_box', 'mass_anno_path_points']].itertuples()):
        try:
            new_basepath = target_path / margin / mass_uid.replace('.', '_')
            record = extract_image_from_dicom(dicom_path, mass_bounding_box, new_basepath, manufacturer, path_points)
            record['mass_uid'] = mass_uid
            record['dicom_path'] = dicom_path
            record['failed'] = False
            records.append(record)
        except Exception as e:
            print(dicom_path, new_basepath)
            records.append({'dicom_path': dicom_path, 'mass_uid': mass_uid, 'failed': e})

    extraction_log = pd.DataFrame(records)
    extraction_log.to_csv(target_path / 'extraction_log.csv')
    return extraction_log


if __name__ == "__main__":

    print("Data augmentation")
    for pos in ["Spiculated","Circumscribed", "Indistinct"]:
        augment_numpy_images(
            path="/usr/xtmp/mammo/npdata/datasetname_with_fa/train/",
            targetNumber=5000,
            targetDir="/usr/xtmp/mammo/npdata/datasetname_with_fa/train_augmented_5000/",
            rot=True,
            with_fa=True)

