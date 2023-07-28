import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from pathlib import Path

import torchvision.transforms as transforms
from dataHelper import DatasetFolder
import torch

from highlighting_precision import hp
import scipy.stats as st

from sklearn.metrics import roc_curve
from scipy.special import softmax
import pyroc

from gradcam_APs import gradcam_aps_main


NUM_BOOTSTRAPS = 3

f_map_location = lambda storage, loc: storage.cuda(torch.cuda.current_device())

def auc_label(auc_obj):
    return f'''AUC: {round(auc_obj.auc[0][0], 2)} ({round(auc_obj.ci()[0][0], 2)}, {round(auc_obj.ci()[1][0], 2)})'''

def age_3_bucket(age):
    if age <= 0:
        return np.nan
    elif age < 50:
        return '<50'
    elif age < 70:
        return '50-69'
    else:
        return '>70'
    
def age_2_bucket(age):
    if age <= 0:
        return np.nan
    elif age <= 57:
        return '<=57'
    else:
        return '>57'

class ExternalValidationContext():

    def __init__(self, environment, external_validation_path, data_path, models, margin_name_labels, malignancy_name_labels, subgroupings, classmap=None):

        self.environment = environment
        
        self.models=models

        self.external_validation_path = Path(external_validation_path)
        self.data_path = Path(data_path)
        self.model_path = self.external_validation_path / 'models'
        self.logit_path = self.external_validation_path / 'logits'

        assert self.external_validation_path.exists()
        assert self.model_path.exists()
        assert self.logit_path.exists()
        assert self.data_path.exists()

        # column metadata
        self.margin_labels = [label for label, _, _ in margin_name_labels]
        self.margin_cols = [col for _, col, _ in margin_name_labels]
        self.margins = [word for _, _, word in margin_name_labels]

        self.label_to_word = {label: word for label, _, word in margin_name_labels}
        self.label_to_col = {label: col for label, col, _ in margin_name_labels}
        self.col_to_label = {col: label for label, col, _ in margin_name_labels}
        self.col_to_word = {col: word for _, col, word in margin_name_labels}
        self.word_to_label = {word: label for label, _, word in margin_name_labels}
        self.word_to_col = {word: col for _, col, word in margin_name_labels}

        # models
        self.vanilla_vgg_path = self.model_path / '0.9504818324659497_at_epoch_60_corrected_format'
        self.protopnet_path = self.model_path / '50_5push0.9209.pth'
        self.iaiabl_path = self.model_path / '50_4push0.9546.pth'
        self.bb_malig_path = self.model_path / '0.8701298701298701_at_epoch_16_corrected_format'

        assert self.vanilla_vgg_path.exists()
        assert self.protopnet_path.exists()
        assert self.iaiabl_path.exists()
        assert self.bb_malig_path.exists()

        self.num_bootstraps = 3

        self.margin_name_labels = margin_name_labels
        self.malignancy_name_labels = malignancy_name_labels

        dataset, dataloader = self.test_dataset_and_loader(data_path)
        self.dataset = dataset
        self.dataloader = dataloader

        self.subgroupings = subgroupings

        self.classmap = classmap

    def __repr__(self):
        return f"ExternalValidationContext(environment={self.environment},\n models={self.models},\n external_validation_path={self.external_validation_path},\n data_path={self.data_path},\n margin_name_labels={self.margin_name_labels},\n malignancy_name_labels={self.malignancy_name_labels},\n vanilla_vgg_path={self.vanilla_vgg_path},\n protopnet_path={self.protopnet_path},\n iaiabl_path={self.iaiabl_path},\n bb_malig_path={self.bb_malig_path},\n subgroupings={self.subgroupings},\n classmap={self.classmap}\n)"

    @classmethod
    def test_dataset_and_loader(cls, data_path):

        dataset = DatasetFolder(
                str(data_path),
                augmentation=False,
                loader=np.load,
                extensions=("npy",),
                transform=transforms.Compose([
                    torch.from_numpy,
                ])
            )
        loader = torch.utils.data.DataLoader(
                dataset, batch_size=8, shuffle=True,
                num_workers=4, pin_memory=False)

        return dataset, loader


    def classwise_activation_precision_for_protopnets(self, num_bootstraps=NUM_BOOTSTRAPS):

        np.random.seed(0)
        torch.random.manual_seed(0)
        ap_ci_df = pd.DataFrame()

        for model_name, m in [('iaia-bl', self.iaiabl_path), ('protopnet', self.protopnet_path)]:
            print(model_name, "-------------")

            res_list_protop = hp(self.data_path, 
                    m,
                    per_proto=True,
                    num_bootstraps=num_bootstraps,
                    verbose=False)


            res_list_protop_np = np.array(res_list_protop)

            circ_indices = res_list_protop_np[:, :, 0] <= 4
            indst_indices = (res_list_protop_np[:, :, 0] > 4) & (res_list_protop_np[:, :, 0] <= 9)
            spic_indices = res_list_protop_np[:, :, 0] > 9

            aps = lambda idxs: np.mean(res_list_protop_np[idxs, 1].reshape(-1, 5), axis=0)

            circ_aps = aps(circ_indices)
            indst_aps = aps(indst_indices)
            spic_aps = aps(spic_indices)

            # create 95% confidence interval
            for ap_list, margin in [(circ_aps, 'circumscribed'), (indst_aps, 'indistinct'), (spic_aps, 'spiculated')]:

                mean_ap = np.mean(ap_list)
                ci = st.t.interval(alpha=0.95, df=len(ap_list)-1,
                    loc=mean_ap,
                    scale=st.sem(ap_list))

                ap_ci_df = pd.concat([ap_ci_df, pd.DataFrame({
                    'ap': [mean_ap],
                    'ci_025': [ci[0]],
                    'ci_975': [ci[1]],
                    'model': [model_name],
                    'margin': [margin],
                    'num_bootstraps': [num_bootstraps]
                })])

        return ap_ci_df


    def margin_predictions_for_models(self, image_has_rois=False):
        np.random.seed(0)
        torch.random.manual_seed(0)
        all_predictions = None

        bb_model_malig = torch.load(self.bb_malig_path, map_location=f_map_location)

        for model_name, model_path in [('protopnet', self.protopnet_path), ('iaia-bl', self.iaiabl_path), ('vgg16', self.vanilla_vgg_path)]:

            print(model_name, "-------------")
            model = torch.load(model_path, map_location=f_map_location)
            model.eval()

            patient_ids = []
            circ = []
            indst = []
            spic = []
            context_sizes = []
            margins = []
            labels = []
            malig_neg = []
            malig_pos = []
            preds = None

            for i, (image, label, patient_id) in tqdm(enumerate(self.dataloader)):
                if 'vgg16' == model_name:
                    if image_has_rois:
                        image = image[:, :-1]
                    pred = model.forward(image.cuda())
                    malig_pred = bb_model_malig(image.cuda())
                else:
                    pred = model(image.cuda())[0]
                    
                for p in range(pred.shape[0]):
                    margin_label = label[p].cpu().item()
                    margins.append(self.label_to_word[margin_label])
                    labels.append(margin_label)
                    circ.append(pred.detach().cpu()[p, 0].item())
                    indst.append(pred.detach().cpu()[p, 1].item())
                    spic.append(pred.detach().cpu()[p, 2].item())
                    patient_ids.append(patient_id[p])
                    context_sizes.append(100)
                    
                    if 'vgg16' == model_name:
                        malig_neg.append(malig_pred.detach().cpu()[p, 0].item())
                        malig_pos.append(malig_pred.detach().cpu()[p, 1].item())

                np_pred = pred.detach().cpu().numpy()
                preds = np_pred if preds is None else np.concatenate([preds, np_pred])

            if 'vgg16' == model_name:
                preds_df = pd.DataFrame({
                    'model_name': model_name,
                    'patient_id': patient_ids,
                    'circ_prob': np.exp(circ),
                    'indst_prob': np.exp(indst),
                    'spic_prob': np.exp(spic),
                    'context_size': context_sizes,
                    'margin': margins,
                    'margin_label': labels,
                    'mal_neg_prob': np.exp(malig_neg),
                    'mal_pos_prob': np.exp(malig_pos)
                })
            else:
                
                probs = softmax(preds, axis=1)

                preds_df = pd.DataFrame({
                    'model_name': model_name,
                    'patient_id': patient_ids,
                    'circ_logit': circ,
                    'indst_logit': indst,
                    'spic_logit': spic,
                    'circ_prob': probs[:,0],
                    'indst_prob': probs[:,1],
                    'spic_prob': probs[:,2],
                    'context_size': context_sizes,
                    'margin': margins,
                    'margin_label': labels
                })
            
            if all_predictions is None:
                all_predictions = preds_df
            else:
                all_predictions = pd.concat([all_predictions, preds_df], sort=True)

        return all_predictions
    
    def roc_figure_for_models(self, predictions_df):
        fig, axs = plt.subplots(2, 3, figsize=(5.0 * 3, 5.0 * 2))

        for j, model in enumerate(self.models):
    
            axs[1][j].set_title(f'{model} ROC\nMalignancy')

            model_df = predictions_df[predictions_df['model_name'] == model]

            for margin_label, col, margin in self.margin_name_labels:
                y_hat = model_df[col]
                
                ax = axs[0][margin_label]
                ax.set_title(f'{col} ROC\n Margin Classification')
                # Margins
                y_as_class = model_df['margin_label'] == margin_label

                margin_roc = pyroc.ROC(y_as_class, y_hat)
                fpr, tpr, thresholds = roc_curve(y_as_class, y_hat)

                ax.plot(fpr, tpr, label=f'{model}, {margin}, {y_as_class.value_counts()[True]} Cases\n     {auc_label(margin_roc)}')
            
            mal_roc = pyroc.ROC(model_df['malignancy_label'], model_df['mal_pos_prob'])
            fpr, tpr, thresholds = roc_curve(model_df['malignancy_label'], model_df['mal_pos_prob'])
            axs[1][j].plot(fpr, tpr, label=f'{model}, {auc_label(mal_roc)}')

        for i in range(3):
            for j in range(2):
                ax = axs[j, i]
                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')
                ax.plot([0, 1], [0, 1],'--', color='gray')
                ax.legend(loc="lower right")

        fig.tight_layout()

    def roc_dfs(self, prediction_df, classification_task, classes, label_col):

        roc_df = pd.DataFrame()
        ci_df = pd.DataFrame()

        for model in self.models:

            # margin roc curves
            scoped_df = prediction_df[prediction_df['model_name'] == model]

            for label, col, clazz in classes:

                ys = scoped_df[label_col] == label
                y_hat = scoped_df[col]

                fpr, tpr, thresholds = roc_curve(ys, y_hat)
                
                this_df = pd.DataFrame({
                    'fpr': fpr,
                    'tpr': tpr,
                    'threshold': thresholds,
                    'model': model,
                    classification_task: clazz
                })

                roc_df = pd.concat([roc_df, this_df])

            # AUCs only
            for subgroup, query in self.subgroupings:

                if subgroup == 'all':
                    queried_df = scoped_df
                elif subgroup in ('manufacturer', 'age_3_bucket', 'age_2_bucket', 'race'):
                    queried_df = scoped_df[scoped_df[subgroup] == query]
                else:
                    raise ValueError(subgroup)

                for label, col, clazz in classes:

                    ys = queried_df[label_col] == label

                    class_counts = ys.value_counts()
                    class_size = 0 if True not in class_counts else class_counts[True]
                    
                    if class_size == 0 or class_size == len(queried_df):
                        ci_df = pd.concat([ci_df, pd.DataFrame({
                            'model': [model],
                            classification_task: [clazz],
                            'group_size': [len(queried_df)],
                            'class_size': [class_size],
                            'subgrouping': [subgroup],
                            'subgroup_query': [query]
                        })], sort=True)

                    else:

                        y_hat = queried_df[col]

                        roc = pyroc.ROC(ys, y_hat)

                        ci_df = pd.concat([ci_df, pd.DataFrame({
                            'auc': roc.auc[0],
                            'ci_025': roc.ci()[0],
                            'ci_975': roc.ci()[1],
                            'model': model,
                            classification_task: clazz,
                            'group_size': len(queried_df),
                            'class_size': class_size,
                            'subgrouping': subgroup,
                            'subgroup_query': query
                        })], sort=True)

        return roc_df, ci_df


    def run_gradcams(self, num_bootstraps=NUM_BOOTSTRAPS, debug_mode=False):
        grad_cam_path = self.external_validation_path / 'grad_cam'
        grad_cam_path.mkdir(exist_ok=True)
        
        aps_all = gradcam_aps_main(test_dir=str(self.data_path),
                    model_path=str(self.vanilla_vgg_path),
                    per_class=True,
                    save_location=str(grad_cam_path/ f'{self.data_path.stem}-tmp.png'),
                    classmap=self.classmap,
                    num_bootstraps=num_bootstraps)

        ci_df = pd.DataFrame()
        for method_margin, ap, ci_lower, ci_upper in aps_all:
            method, margin = method_margin.lower().split('class')
            ci_df = pd.concat([ci_df, pd.DataFrame({
                'ap': [ap],
                'ci_025': [ci_lower],
                'ci_975': [ci_upper],
                'method': [method.strip()[:-3]],
                'margin': [self.label_to_word[int(margin.strip())]],
                'model': ['vgg16'],
                'num_bootstraps': [num_bootstraps]
            })])

        return ci_df

    def patient_summary_statistics(self, df):
        # Assuming predictions_wmal is already defined
        __df = df[df['model_name'] == 'iaia-bl']

        # Initialize an empty list to store our data
        data = []

        # Add unique counts
        data.append({'subgroup': 'unique_patients', 'value': 'total', 'count': __df['patient_id'].nunique()})
        data.append({'subgroup': 'unique_lesions', 'value': 'total', 'count': __df['mass_uid'].nunique()})
        data.append({'subgroup': 'unique_images', 'value': 'total', 'count': __df['image_id'].nunique()})

        # Add patient age range
        data.append({'subgroup': 'patient_age_range', 'value': 'min', 'count': __df['age'].min()})
        data.append({'subgroup': 'patient_age_range', 'value': 'max', 'count': __df['age'].max()})

        # Process each column for counts
        for col in {g[0] for g in self.subgroupings if g[0] != 'all'}:
            counts = __df.groupby(col)['patient_id'].count().reset_index()
            counts.rename(columns={'patient_id': 'count', col: 'value'}, inplace=True)
            counts['subgroup'] = col
            data.extend(counts.to_dict('records'))

        # Create the consolidated DataFrame
        return pd.DataFrame(data)
    
    def plot_margin_probs(self, df):
        figure, axs = plt.subplots(1,3, figsize=(15, 4), sharey=True, sharex=True)
        for j, model in enumerate(self.models):
            for i, clazz in enumerate(['circ_prob', 'indst_prob', 'spic_prob']):
                ax = axs[i]

                ax.hist(df[df['model_name'] == model][clazz], label=model, alpha=.6)
                ax.set_title(f'Histogram of {clazz} scores')
                ax.set_xlabel(f'{clazz} score')
                ax.set_ylabel('Frequency')
                ax.legend()
                
        plt.show()

    @classmethod
    def malignancy_for_models(slf, prediction_df):

        def sigmoid(z):
            return 1/(1 + np.exp(-z))

        def mal_prob_for_weights(w):
            
            def mal(row):
                y_hat = w['circ'] * row.circ_logit + w['indst'] * row.indst_logit + w['spic'] * row.spic_logit
                prob = sigmoid((y_hat + w['intercept']))
                return prob

            return mal

        protopnet_malignancy_weights = {
            'circ': 0.04777971,
            'indst': -0.0171781,
            'spic': .32511213,
            'intercept': 1.71309941
        }

        iaiabl_maligancy_weights = {
            'circ': -.16543867,
            'indst': -.1031895,
            'spic': .5866863,
            'intercept': -1.55195357
        }

        __df = prediction_df.copy()
        for model, weights in [('iaia-bl', iaiabl_maligancy_weights), ('protopnet', protopnet_malignancy_weights)]:
            __df.loc[__df['model_name'] == model, 'mal_pos_prob'] = __df.loc[__df['model_name'] == model].apply(mal_prob_for_weights(weights), axis=1)

        __df.loc[__df['mal_neg_prob'].isna(), 'mal_neg_prob'] = __df['mal_pos_prob'].apply(lambda x: 1-x)
        __df['malignancy_pred'] = __df['mal_pos_prob'] > __df['mal_neg_prob']
        return __df