import csv
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#import kaplanmeier as km
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from scipy.stats import entropy as shannon_entropy


cell_types = ["CD4",  "CD20", "CD8", "FoxP3", "CD4_Single", "CD8_Single", "CD4_Treg", "CD8_Treg", "B_cells", "PanCKsingle"]
colors = {"CD4_Treg":'pink',  "B_cells":'blue', "CD8_Treg":'yellow', "CKSingle":'black', "Stroma":'purple', "Tumor":'black', "CD4_Single": "red", "CD8_Single": "green"}



cell_phenos = ["CD4_Single", "CD8_Single", "CD4_Treg", "CD8_Treg", "B_cells", "CKSingle"]


def plot_phenos(cells_data, sample_name, plot_min, plot_max):
    plt.figure(figsize=(10,10))
    plt.xlim((plot_min-50, plot_max+50))
    plt.ylim((plot_min-50, plot_max+50))
    
    for cell_type in cell_phenos:
        cells_data_type = cells_data.loc[cells_data[cell_type] == 1]
            
        xs = cells_data_type['x'].to_numpy()
        ys = cells_data_type['y'].to_numpy()
        if ys.shape[0] > 0:
            ys = np.amax(ys) + np.amin(ys) - ys
        plt.scatter(xs, ys, color=colors[cell_type], label=cell_type, s=3)
    
    plt.legend()
    plt.savefig("cell_plots/" + str(sample_name) + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_tissue(cells_data, sample_name, plot_min, plot_max, label, test=False):
    plt.figure(figsize=(20, 20), dpi=50)
    plt.xlim(plot_min - 50, plot_max + 50)
    plt.ylim(plot_min - 50, plot_max + 50)
    
    # Create color map
    colors = {0: "green", 1: "brown"}
    labels = {0: "Stroma", 1: "Tumor"}
    
    # Single scatter plot with color based on "Cancer" column
    plt.scatter(
        cells_data['x'], 
        cells_data["y"], 
        c=cells_data["Cancer"].map(colors),
        label=cells_data["Cancer"].map(labels),
        alpha=0.7,
        s=10
    )
    # Create a custom legend
    handles = [plt.scatter([], [], color=colors[i], label=labels[i]) for i in colors]
    plt.legend(handles=handles)
    plt.title("Cell Distribution by Tissue Type")
    
    if not test:
        word = label
        plt.text(10, 10, word, fontsize=34)
        plt.savefig("cell_plots/tissue_plots/" + str(sample_name) + "_tissue_type_seg"  + ".png", bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig("cell_plots/test_plots/" + str(sample_name) + "_tissue_type_seg"  + ".png", bbox_inches='tight', pad_inches=0)
    plt.close()

        
def plot_cells(cells_data_all, samples_data, patient_data, test_ids):
    print(test_ids)
    xs = cells_data_all['x'].to_numpy()
    ys = cells_data_all['y'].to_numpy()
    xmax = np.amax(xs)
    ymax = np.amax(ys)
    plot_max = np.amax((xmax,ymax))
    xmin = np.amin(xs)
    ymin = np.amin(ys)
    plot_min = np.amin((xmin,ymin))
    test_data = {"ID":[], "sample":[], "label":[]}
    for ID in patient_data["ID"]:
        samples = samples_data[samples_data["ID"] == ID]
        
        label = patient_data[patient_data["ID"] == ID]["Tumor_type_code"].values[0]
        for sample_name in samples['sample_name_simple']:
            print("Making plot for {}".format(sample_name))
            
        
            cells_data = cells_data_all.loc[cells_data_all['Sample Name'] == sample_name]

            #plot_phenos(cells_data, sample_name, plot_min, plot_max)
            
            plot_tissue(cells_data, sample_name, plot_min, plot_max, label, ID in test_ids)
            if ID in test_ids:
                print("test ID")
                test_data["ID"].append(ID)
                test_data["sample"].append(sample_name)
                test_data["label"].append(label)

    pd.DataFrame(test_data).to_csv("cell_plots/test_data.csv", index=False)

def impute_missing_values(df, strategy='mean', fill_value=None):
    """
    Impute missing values in a DataFrame.

    Parameters:
    - df: pandas DataFrame with NaN values to impute.
    - strategy: The imputation strategy to use. Options: 'mean', 'median', 'most_frequent', 'constant'.
    - fill_value: The constant value to use if the strategy is 'constant'. Default is None.

    Returns:
    - DataFrame with missing values imputed.
    """
    # Check if the strategy is 'constant' and fill_value is provided
    if strategy == 'constant' and fill_value is None:
        raise ValueError("For 'constant' strategy, you must provide a fill_value.")
    
    # Create an imputer object based on the chosen strategy
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    
    # Impute missing values
    df_imputed = df.copy()  # Make a copy to avoid modifying the original
    df_imputed[:] = imputer.fit_transform(df_imputed)
    
    return df_imputed


def compute_shannon_entropy(series):
    # Get the value counts and normalize to get probabilities
    value_counts = series.value_counts(normalize=True)
    
    # Compute Shannon entropy
    entropy = shannon_entropy(value_counts)#-np.sum(value_counts * np.log2(value_counts))
    
    return entropy




def preprocess_cells(data):
    #data["ID"] = data["ID"].map(lambda x: int(x.replace("Lung # ", "")))
    #data = data.drop('Cell ID', axis=1)
    data["Tissue Category"] = data["Tissue Category"].map(lambda x: 1 if x == "Tumor" else 0)
    
    data["Cell X Position"] = data["Cell X Position"].map(lambda x: float(x) )
    data["Cell Y Position"] = data["Cell Y Position"].map(lambda x: float(x) )
    data = data.rename(columns={
                                #"Nucleus Opal 620 Mean (Normalized Counts, Total Weighting)": "FoxP3",
                                "Entire Cell Opal 520 Mean (Normalized Counts, Total Weighting)": "CD4",
                                "Entire Cell Opal 540 Mean (Normalized Counts, Total Weighting)": "CD20",
                                "Entire Cell Opal 570 Mean (Normalized Counts, Total Weighting)": "CD8",
                                "Entire Cell Opal 650 Mean (Normalized Counts, Total Weighting)": "CD45RO",
                                "Entire Cell Opal 690 Mean (Normalized Counts, Total Weighting)": "PanCK",
                                "Nucleus DAPI Mean (Normalized Counts, Total Weighting)": "DAPI",
                                "Entire Cell Autofluorescence Mean (Normalized Counts, Total Weighting)": "Autofluorescence",
                                "Nucleus Area (pixels)": "Nucleus Area"
                                })
    column_names = ["Cancer", "CD4", "CD20", "CD8", "PanCK", "DAPI", "Nucleus Area", "Nucleus Compactness",
                    "Nucleus Axis Ratio", "CD4_Single", "CD4_Treg", "CD8_Single", "CD8_Treg", "B_cells", "CKSingle", "Stroma_other"]
    
    data.rename(columns = {'Tissue Category':'Cancer', 'Cell X Position': 'x', 'Cell Y Position': 'y'}, inplace = True)

    for column in tqdm(column_names):
        data[column] = data[column].astype(float)
        #data[column] = data[column].map(lambda x: float(x))



    data["CK_Single"] = ((data['CK'] == 1) & (data['CD4_Single'] == 0) & (data['CD4_Treg'] == 0)
                         & (data['CD8_Single'] == 0) & (data['CD8_Treg'] == 0)
                         & (data['B_cells'] == 0)
                         ).astype(int)
        
    data = data.drop('CD45RO', axis=1)
    data = data.drop('Lab ID', axis=1)
    data = data.drop('inForm 2.4.6781.17769', axis=1)
    print("num cells before dropping na", len(data))
    
    data = data[data.notna().all(axis=1)]
    print(data.isna().any())
    print("num cells after dropping na", len(data))
    return data





def preprocess_patients(data):
    data = data.rename(columns={"ID or PAD_year":"ID", "Tumor_type":"Histology", "Sex":"Gender" , "Event_last_followup": "Dead/Alive"})
    t = {"Adenocarcinoma": "LUAD", "Squamous cell carcinoma": "LUSC", "Other":"Other"}
    data["Tumor_type_code"] = data["Histology"].map(lambda x: t[x])
    data["Smoking"] = data["Smoking"].map(lambda x: 0 if x == "Never-smoker" else 1)
    data["LUAD"] = (data["Tumor_type_code"] == "LUAD").astype(int)
    data["Gender"] = (data["Gender"] == "Male").astype(int)
    #data["Performance status (WHO)"] = data["Performance status (WHO)"].map(lambda x: 1 if x > 0 else 0)
    stage_dict = {"Ia": 0, "Ib": 1, "IIa": 2, "IIb": 3, "IIIa": 4, "IIIb": 5, "IV": 6}
    data["Stage"] = data["Stage (7th ed.)"].map(lambda x: stage_dict[x])
    
    return data[["ID", "LUAD", "Age", "Gender", "Smoking", "Stage", "Performance status (WHO)", "Follow-up (days)", "label", "censored", "Tumor_type_code"]]



def filter_overlapping(patient_data, samples, cell_data):
    
    patient_ids  = patient_data["ID"]
    print(len(cell_data))
    samples_ids = pd.unique(samples["ID"])
    samples["sample_name_simple"] = samples["sample_name"].map(lambda x: x[:x.rfind("_")])
    samples["sample_name_simple"] = samples["sample_name_simple"].map(lambda x: x.replace("Core[1,", "["))
    sample_names = samples["sample_name_simple"].unique()

    cell_data = cell_data[cell_data["Sample Name"].isin(sample_names)]
    print("num cells samples", len(cell_data))
    cells_ids = pd.unique(cell_data["ID"])


    
    ids = np.intersect1d(patient_ids, samples_ids)
    
    ids = np.intersect1d(cells_ids, ids)
    
    return patient_data[patient_data["ID"].isin(ids)], samples[samples["ID"].isin(ids)], cell_data[cell_data["ID"].isin(ids)]



            

def get_density_morphology_data(data, patient_data, samples_data):
    IDs = pd.unique(data["ID"])
    

    clinical_names = ["label", "LUAD", "Age", "Gender", "Smoking", "Stage", "Performance status"]
    
    statistics = {"ID": [], "time": [], "event": []}
    for n in clinical_names:
        statistics[n] = []

    pheno_names = ["CD4_Single", "CD4_Treg", "CD8_Single", "CD8_Treg", "B_cells"]#, "CK_Single", "Stroma_other"]
    #pheno_names = ["CD4_Single", "CD4_Treg", "CD8_Single", "CD8_Treg", "B_cells", "CK_Single"]
    nuclei_names = ["Nucleus Axis Ratio", "Nucleus Compactness", "Nucleus Area"]

    """
    for phenotype in pheno_names:
        for nuclei_feature in nuclei_names:
            statistics[phenotype + " " + nuclei_feature] = []
    """
    """
    for region in ["Stroma", "Tumor"]:
        for feature_name in nuclei_names:
            for param in ["mean", "std"]:
                statistics[region + " " + feature_name +  " " + param] = []
    """
    """
    for region in ["Stroma", "Tumor"]:
        for feature_name in nuclei_names:
            statistics[region + " " + feature_name] = []
    """
    
    for feature_name in nuclei_names:
        statistics[feature_name] = []
        #statistics[feature_name + " mean"] = []
        #statistics[feature_name + " std"] = []
    

        
    for region in ["Stroma", "Tumor"]:
        for feature_name in pheno_names:
            statistics[region + " " + feature_name] = []

    
    

    for ID in tqdm(IDs):
        sample = samples_data[samples_data["ID"] == ID]
        patient = patient_data[patient_data["ID"]==ID]
        t = patient["Follow-up (days)"].values[0]
        event = 1 if patient["censored"].values[0] else 0
        label = patient["label"].values[0]
        luad = patient["LUAD"].values[0]
        #lusc = patient["LUSC"].values[0]
        age = patient["Age"].values[0]
        gender = patient["Gender"].values[0]
        smoking = patient["Smoking"].values[0]
        stage = patient["Stage"].values[0]
        performance = patient["Performance status (WHO)"].values[0]
        statistics["ID"].append(ID)
        statistics["time"].append(t)
        statistics["event"].append(event)
        statistics["label"].append(label)
        statistics["LUAD"].append(luad)
        #statistics["LUSC"].append(lusc)
        statistics["Age"].append(age)
        statistics["Gender"].append(gender)
        statistics["Smoking"].append(smoking)
        statistics["Stage"].append(stage)
        statistics["Performance status"].append(performance)

        
        patient_cells = data[data["ID"] == ID].copy()

        #for nuclei_feature in nuclei_names:
        #    #statistics[nuclei_feature].append(patient_cells[nuclei_feature].std()/patient_cells[nuclei_feature].mean())
        #    statistics[nuclei_feature].append(patient_cells[nuclei_feature].mean())

        """
        for phenotype in pheno_names:
            phenotype_cells = patient_cells[patient_cells[phenotype] == 1].copy()
            for nuclei_feature in nuclei_names:
                if len(phenotype_cells[nuclei_feature]) <= 1:
                    print(phenotype)
                    #statistics[phenotype + " " + nuclei_feature].append(data[data[phenotype] == 1].copy()[nuclei_feature].std())
                    statistics[phenotype + " " + nuclei_feature].append(0)
                else:
                    #statistics[phenotype + " " + nuclei_feature].append(phenotype_cells[nuclei_feature].mean())
                    statistics[phenotype + " " + nuclei_feature].append(
                        phenotype_cells[nuclei_feature].std()/phenotype_cells[nuclei_feature].mean())
        """
 
        
        for region in ["Stroma", "Tumor"]:
            sample_region_name = "tumor_area" if region=="Tumor" else "stroma_area"
            region_cells = patient_cells[patient_cells["Cancer"] == (region=="Tumor")]
            region_area = sample[sample_region_name].sum()
            for feature_name in pheno_names:
                statistics[region + " " + feature_name].append(region_cells[feature_name].sum()/region_area)
        
        
        #ck_cells = patient_cells[patient_cells["CK_Single"] == 1]
        ck_cells = patient_cells[patient_cells["Cancer"] == 1]
        for nuclei_feature in nuclei_names:
            statistics[nuclei_feature].append(ck_cells[nuclei_feature].std()/ck_cells[nuclei_feature].mean())
        """if len(ck_cells) <= 1:
            for nuclei_feature in nuclei_names:
                statistics[nuclei_feature].append(0)
                #statistics[nuclei_feature + " mean"].append(0)
                #statistics[nuclei_feature + " std"].append(0)
        else:
            for nuclei_feature in nuclei_names:
                statistics[nuclei_feature].append(ck_cells[nuclei_feature].std())
                #statistics[nuclei_feature + " mean"].append(ck_cells[nuclei_feature].mean())
                #statistics[nuclei_feature + " std"].append(ck_cells[nuclei_feature].std())
                #statistics[nuclei_feature].append(ck_cells[nuclei_feature].std()/ck_cells[nuclei_feature].mean())
        """
        
        """
        for region in ["Stroma", "Tumor"]:
            region_cells = patient_cells[patient_cells["Cancer"] == (region=="Tumor")]
            for nuclei_feature in nuclei_names:
                statistics[region + " " + nuclei_feature].append(region_cells[nuclei_feature].mean())
        """
        """for region in ["Stroma", "Tumor"]:
            region_cells = patient_cells[patient_cells["Cancer"] == (region=="Tumor")]
            for nuclei_feature in nuclei_names:
                for param in ["mean", "std"]:
                    if param == "mean":
                        value = region_cells[nuclei_feature].mean()
                    else:
                        value = region_cells[nuclei_feature].std()
                
                    statistics[region + " " + nuclei_feature + " " + param].append(value)"""
        
    for key in statistics:
        print(key, len(statistics[key]))
        
    statistics = pd.DataFrame(statistics)
    
    #statistics[nuclei_names] = impute_missing_values(statistics[nuclei_names])
    return statistics

cells_path = "to_start_with/BOMI2_all_cells_TIL.csv"

samples_path = "../multiplex_dataset/lung_cancer_BOMI2_dataset/samples.csv"#"to_start_with/Lung_TIL_samples.csv"
patients_path = "../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_survival_prediction/Clinical_data_with_labels.csv"#"../lung_cancer_dataset/raw_data/Clinical_data_max_20190329.csv"#"to_start_with/Clinical_data.xlsx"#"to_start_with/BOMI2.xlsx"
densities_path = "patient_cell_density_data_2024.csv"
morphologies_path = "patient_cell_statistics_morphology.csv"
idalign_path = "id_align.csv"

print("loading data")

cells_data = pd.read_csv(cells_path)
samples_data = pd.read_csv(samples_path)
patient_data = pd.read_csv(patients_path)




print("preprocessing data")

cells_data = preprocess_cells(cells_data)

patient_data = preprocess_patients(patient_data)


print(len(patient_data))
patient_data, samples_data, cells_data = filter_overlapping(patient_data, samples_data, cells_data)
print("number of patients after filtering overlapping:", len(patient_data["ID"]))



patient_cells_data = get_density_morphology_data(cells_data, patient_data, samples_data)
patient_cells_data.to_csv("patient_densities_morphologies.csv", index=False)

#print(patient_cells_data)



test_ids = pd.read_csv("../multiplex_dataset/lung_cancer_BOMI2_dataset/binary_subtype_prediction_ACvsSqCC/static_split/test.csv")["ID"].values
#plot_cells(cells_data, samples_data, patient_data, test_ids)




fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
df = cells_data[cells_data["CK_Single"] == 1]
print(len(df))#df = cells_data
ax.scatter(df["Nucleus Axis Ratio"], df["Nucleus Compactness"], df["Nucleus Area"], s=1, alpha=0.5)
ax.set_xlabel('axis ratio')
ax.set_ylabel('compactness')
ax.set_zlabel('area')


plt.show()
