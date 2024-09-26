#!/usr/bin/env python3
from pathlib import Path
import json
import time
from time import sleep
from ml_loop.ml_modules import *
from ml_loop.ml_utils import *
# from ml_loop.ml_utils import tecan_proc, tecan_read
from ml_loop.set_transfomer import SmallSetTransformer_v2
# from tools.globus_batman2chemspeed import *
# from tools.flow_tecan import tecan_flow
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import csv
import matplotlib.pyplot as plt
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def plot_color_circle(hex_color):    
    fig, ax = plt.subplots()
    circle = Circle((0.5, 0.5), 0.4, color=hex_color)
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()


from PIL import Image, ImageDraw

def create_colored_circles(hex_colors, diameter=100, spacing=20):
    # Create a new blank image with a white background
    width = diameter
    height = diameter * len(hex_colors) + spacing * (len(hex_colors) - 1)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Draw and fill circles
    for i, color in enumerate(hex_colors):
        top_left = (0, i * (diameter + spacing))
        bottom_right = (diameter, i * (diameter + spacing) + diameter)
        draw.ellipse([top_left, bottom_right], fill=color)

    return image


def get_hex_color_single(Lab_list):
    color_list = []

    lab = LabColor(lab_l = Lab_list[0], lab_a = Lab_list[1], lab_b = Lab_list[2], observer='2', illuminant='d65')
    rgb = convert_color(lab, sRGBColor)
    rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
 
    hex_color = rgb.get_rgb_hex() 
    return hex_color

def get_hex_color(L, a, b, df=None):
    color_list = []
    y_pred = list(zip(L, a,b))

    for i in range(len(y_pred)):
        lab_list = (list(y_pred[i]))
        a = ', '.join(str(item) for item in list(y_pred[i]))
        lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
        rgb = convert_color(lab, sRGBColor)
        rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
        try:
            c = rgb.get_rgb_hex()
            color_list.append(str(c))
        except:
            color_list.append('#000000')
    try:
        df['color_shade'] = color_list
    except:
        pass
    return df, color_list

df_known = pd.read_csv('C:/Users/kvriz/Desktop/polybot_workcell/ml_electrochromics_database_plus_exp.csv') #
#df_random['L* (Colored State)'] = np.repeat(L_value, df_random.shape[0])
df_known , random_color_list = get_hex_color(df_known['L* (Colored State)'], df_known['a* (Colored State)'], df_known['b*(Colored State)'], df_known)
print(df_known.shape, len(random_color_list))

# tested_candidates = pd.read_csv('all_results.csv')
# # print(tested_candidates)
# iteration=0

# # 3. Call the pretrained ML model , ML precidt the L,a,b values of the provided file
# device ='cpu'
# epochs, learning_rate, batch_size, Ndims = 100, 1e-3, 12, 1056 
# dropout_ratio = 0.2  # replace with your desired value
# model=SmallSetTransformer_v2(dropout_ratio, device, epochs, learning_rate, batch_size)

# # # load the model which was pretrained on the literature data + inhouse samples
# ckpt = torch.load(os.path.join('polybot_app/ml_loop/set_transformer_checkpoints', f'set_transformer_dft_ecfp_0.tar'))
# model.load_state_dict(ckpt['state_dict'])

# # # local_path_from_tecan = "/home/rpl/workspace/polybot_workcell/polybot_app/demo_files/from_tecan"
# # # fname_tecan = f"ECP_demo_batch_{iteration}.asc" 
# # # filename=os.path.join(local_path_from_tecan, fname_tecan)

# # # df = tecan_read(filename)
# # # lab_values = tecan_proc(df)
# # # print('lab_values',lab_values)
# lab_values = tested_candidates[['L', 'a', 'b']]


# y_train =  lab_values
# y_train =  np.array(y_train, dtype=np.float16)
# # 10. Retrain the ML model and save the new weights to the checkpoints folder  
# X_train, y_train = get_train_data_representation_dft(tested_candidates) , lab_values
# print(y_train)
# train_data_1, train_data_2,train_data_3, y_train = np.array(X_train.iloc[:, :Ndims].values, dtype=np.float16),  np.array(X_train.iloc[:, Ndims:2*Ndims].values, dtype=np.float16), np.array(X_train.iloc[:, 2*Ndims:].values, dtype=np.float16), np.array(y_train, dtype=np.float16)
# y_train = pd.DataFrame(y_train, columns=['L', 'a', 'b'])

# my_scaler = joblib.load('polybot_app/ml_loop/scaler.gz')
# y_train = my_scaler.transform(y_train.values)\

# model.train_model(train_data_1, train_data_2,train_data_3, y_train)    
# torch.save({'state_dict':model.state_dict()}, os.path.join('polybot_app/ml_loop/set_transformer_checkpoints', f"set_transformer_dft_ecfp_2.tar"))
    

def iter_run(init_file, iteration):
        
    """Actions to perform to run one full experimental loop:
    The experiment starts when the user decides on the color,i.e., Lab values, of the polymer they want to synthezise 
    and setting the starting monomers they have in the inventory
    """

    # Read the init.json to extract the user provided information and store the metadata
    with open(init_file, 'r') as f:
         init_data = json.load(f)

    # 3. Call the pretrained ML model , ML precidt the L,a,b values of the provided file
    device ='cpu'
    epochs, learning_rate, batch_size, Ndims = 5, 1e-3, 12, 1056
    dropout_ratio = 0.15
    model=SmallSetTransformer_v2(dropout_ratio, device, epochs, learning_rate, batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Initialize optimizer

    # load the model which was pretrained on the literature data + inhouse samples
    ckpt = torch.load(os.path.join('polybot_app/ml_loop/set_transformer_checkpoints', f'set_transformer_new_model_0_test_12_400_02_final.tar')) #
    #set_transformer_dft_ecfp_2_100_new.tar

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    # 4. Use the Set transformer model to get all the predictions
    all_possibilities_avail = all_possibilities[all_possibilities.exclude==0]
 
    lab_scaler = joblib.load('C:/Users/kvriz/Desktop/polybot_workcell/polybot_app/ml_loop/scaler.gz')
    X_valid= all_possibilities_avail.iloc[:,3:-1]
    pred_data_1, pred_data_2,pred_data_3, y_target = np.array(X_valid.iloc[:, :Ndims].values, dtype=np.float16),  np.array(X_valid.iloc[:, Ndims:2*Ndims].values, dtype=np.float16), np.array(X_valid.iloc[:, 2*Ndims:].values, dtype=np.float16), np.zeros(X_valid.shape[0])
    preds, uncertainties = model.test_model(pred_data_1, pred_data_2,pred_data_3, y_target)
    
  
    preds_inv_scaled = lab_scaler.inverse_transform(preds)
    #target_Lab = init_data["target_Lab"]
    target_Lab = [75, 36, 70] #[75, 40, 75] # [75, 40, 70]
    print('target_Lab',target_Lab )
    #hex_target_color = get_hex_color_single(target_Lab)
    #print('hex_target_color', hex_target_color)
    #plot_color_circle(hex_target_color)
    preds_df = pd.concat([pd.DataFrame(all_possibilities_avail['smiles1'].values, columns=["smiles1"]),
                     pd.DataFrame(all_possibilities_avail['percentage_1'].values, columns=["percentage_1"]),
                     pd.DataFrame(all_possibilities_avail['smiles2'].values, columns=["smiles2"]),
                     pd.DataFrame(all_possibilities_avail['percentage_2'].values, columns=["percentage_2"]),
                     pd.DataFrame(all_possibilities_avail['smiles3'].values, columns=["smiles3"]),
                     pd.DataFrame(all_possibilities_avail['percentage_3'].values, columns=["percentage_3"]),
                     pd.DataFrame(preds_inv_scaled, columns=["L", "a", "b"]) ], axis=1)
    #preds_df.to_csv('all_predictions.csv', index=None)
    # create a plot to visualize the color preds_df
    
    y_pred = list(zip(preds_df['L'], preds_df['a'], preds_df['b']))
    color_list = []

    for i in range(len(y_pred)):
            lab_list = (list(y_pred[i]))
            a = ', '.join(str(item) for item in list(y_pred[i]))
            lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
            rgb = convert_color(lab, sRGBColor)
            rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
            try:
                c = rgb.get_rgb_hex()
                color_list.append(str(c))
            except:
                color_list.append('#000000')
    
    # image = create_colored_circles(color_list)
    # image.show()
    # df_known , random_color_list = get_hex_color(df_known['L* (Colored State)'], df_known['a* (Colored State)'], df_known['b*(Colored State)'],
    
    
    # 5. Select: get the 6 top candidates to run in Chemspeed: create the correct format and send it to chemspeed, file named 
    # according to the experimental iteration
    top_candidates = select_next_exp_ml(preds_df, target_Lab, uncertainties, iteration) # print out the first monomer smiles + ratio as well
    print('top_candidates', top_candidates)
    # top_candidates.to_csv('loop-2_new_suggested_exp_2024.csv', index=None)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.axvline(x=0)
    ax.axhline(y=0)
    ax.scatter(target_Lab[1], target_Lab[2], marker='*', edgecolors='black', s=150,color='w',
             linewidths=1.5)
    print(df_known['a* (Colored State)'].shape,  df_known['b*(Colored State)'].shape, len(random_color_list))
    ax.scatter(df_known['a* (Colored State)'], df_known['b*(Colored State)'], c=random_color_list, label='database', marker='^')
    # ax.scatter(preds_df['a'], preds_df['b'], c=color_list, s=80, label='loop_1')
    
    # ax.scatter(top_candidates.a, top_candidates.b, color='r',marker='s',s=100,facecolor='None')
    # for i, txt in enumerate(['AI-1','AI-2','AI-3','AI-4','AI-5','AI-6']):
    #     ax.annotate(txt, (top_candidates.a.values[i], top_candidates.b.values[i]))
    # plt.show()
    y_pred = list(zip(top_candidates['L'], top_candidates['a'], top_candidates['b']))
    color_list = []

    for i in range(len(y_pred)):
            lab_list = (list(y_pred[i]))
            a = ', '.join(str(item) for item in list(y_pred[i]))
            lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
            rgb = convert_color(lab, sRGBColor)
            rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
            try:
                c = rgb.get_rgb_hex()
                color_list.append(str(c))
            except:
                color_list.append('#000000')
    # image = create_colored_circles(color_list)
    # image.show()

    experimental_loop_1 = pd.read_csv('polybot_app/demo_files/metadata/test1.csv')
    # print(experimental_loop_1)
    y_pred = list(zip(experimental_loop_1['L_exp'],experimental_loop_1['a_exp'], experimental_loop_1['b_exp']))
    color_list = []

    for i in range(len(y_pred)):
            lab_list = (list(y_pred[i]))
            a = ', '.join(str(item) for item in list(y_pred[i]))
            lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
            rgb = convert_color(lab, sRGBColor)
            rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
            try:
                c = rgb.get_rgb_hex()
                color_list.append(str(c))
            except:
                color_list.append('#000000')
    ax.scatter(experimental_loop_1.a_exp, experimental_loop_1.b_exp, c=color_list,s=100,marker='o',facecolor='black',edgecolors='black', label='Loop-1')#
    # for i, txt in enumerate(['EXP-1','EXP-2','EXP-3','EXP-4','EXP-5','EXP-6']):
    #     ax.annotate(txt, (experimental_loop_1.a_exp.values[i], experimental_loop_1.b_exp.values[i]))
    ax.set_xlabel('a* (Colored State)', fontsize=14)
    ax.set_ylabel('b* (Colored State)', fontsize=14)
    # image = create_colored_circles(color_list)
    # image.show()


    experimental_loop_2 = pd.read_csv('polybot_app/demo_files/metadata/test2.csv')
    # print(experimental_loop_1)
    y_pred = list(zip(experimental_loop_2['L_pred'],experimental_loop_2['a_pred'], experimental_loop_2['b_pred']))
    color_list = []

    for i in range(len(y_pred)):
            lab_list = (list(y_pred[i]))
            a = ', '.join(str(item) for item in list(y_pred[i]))
            lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
            rgb = convert_color(lab, sRGBColor)
            rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
            try:
                c = rgb.get_rgb_hex()
                color_list.append(str(c))
            except:
                color_list.append('#000000')
    ax.scatter(experimental_loop_2.a_exp, experimental_loop_2.b_exp, c=color_list,s=100,marker='s',facecolor='black', edgecolors='black', label='Loop-2')#
    # for i, txt in enumerate(['EXP-1','EXP-2','EXP-3','EXP-4','EXP-5','EXP-6']):
    #     ax.annotate(txt, (experimental_loop_2.a_exp.values[i], experimental_loop_2.b_exp.values[i]))
    ax.set_xlabel('a* (Colored State)', fontsize=14)
    ax.set_ylabel('b* (Colored State)', fontsize=14)
    # image = create_colored_circles(color_list)
    # image.show()


    experimental_loop_3 = pd.read_csv('polybot_app/demo_files/metadata/test3.csv')
    # print(experimental_loop_1)
    y_pred = list(zip(experimental_loop_3['L_pred'],experimental_loop_1['a_pred'], experimental_loop_3['b_pred']))
    color_list = []

    for i in range(len(y_pred)):
            lab_list = (list(y_pred[i]))
            a = ', '.join(str(item) for item in list(y_pred[i]))
            lab = LabColor(lab_l = lab_list[0], lab_a = lab_list[1], lab_b = lab_list[2], observer='2', illuminant='d65')
            rgb = convert_color(lab, sRGBColor)
            rgb = colormath.color_objects.sRGBColor(rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b, is_upscaled=False)
            try:
                c = rgb.get_rgb_hex()
                color_list.append(str(c))
            except:
                color_list.append('#000000')
    ax.scatter(experimental_loop_3.a_exp, experimental_loop_3.b_exp, c=color_list,s=100,marker='d',facecolor='black', edgecolors='black',label='Loop-3')#
    # for i, txt in enumerate(['EXP-1','EXP-2','EXP-3','EXP-4','EXP-5','EXP-6']):
    #     ax.annotate(txt, (experimental_loop_3.a_exp.values[i], experimental_loop_3.b_exp.values[i]))
    ax.set_xlabel('a* (Colored State)', fontsize=14)
    ax.set_ylabel('b* (Colored State)', fontsize=14)
    ax.set_zlabel('L* (Colored State)', fontsize=14)
    plt.legend(fontsize=14)
    # image = create_colored_circles(color_list)
    # image.show()

    # plt.savefig('loop-1_color_wheel.svg')

    plt.show()


init_file = 'polybot_app/demo_files/init.json'

with open(init_file, 'r') as f:
    init_data = json.load(f)

# 2. Given the initial file with the inventory generate all the possible combinations
inventory = init_data["monomers"]
num_monomers = init_data["number_of_monomers"]
ratio_step = init_data["ratio_step"]



all_possibilities = create_predictions_dataset_dft(inventory, num_monomers, ratio_step) #, number if monomers and step 
# print('shape of space', all_possibilities.shape)
#all_possibilities = all_possibilities.drop(indices_rem  ).reset_index().drop(columns = ['index'])
# all_possibilities=pd.read_csv('all_possible_space.csv') 
all_possibilities['percentage_1'] = all_possibilities['percentage_1']/100
all_possibilities['percentage_2'] = all_possibilities['percentage_2']/100
all_possibilities['percentage_3'] = all_possibilities['percentage_3']/100
dataset = all_possibilities[all_possibilities["percentage_1"] !=1]
dataset=dataset.reset_index(drop=True)
validation_1 = bits_to_df(dataset['smiles1'], 'bit_1')
validation_2 = bits_to_df(dataset['smiles2'], 'bit_2')
validation_3 = bits_to_df(dataset['smiles3'], 'bit_3')
df1_dft = dft_descr_from_df(dataset['smiles1'], 'A')
df2_dft  = dft_descr_from_df(dataset['smiles2'], 'B')
df3_dft  = dft_descr_from_df(dataset['smiles3'], 'C')

validation_dataset = pd.concat([
    dataset['smiles1'],
    dataset['smiles2'],
    dataset['smiles3'],
    validation_1,df1_dft,
    dataset[['percentage_1']],
    validation_2,df2_dft,
    dataset[['percentage_2']],
    validation_3, df3_dft,
    dataset[['percentage_3']]
], axis=1)      
all_possibilities = validation_dataset

print(all_possibilities.shape)
df_known = pd.read_csv('C:/Users/kvriz/Desktop/polybot_workcell/ml_electrochromics_database_plus_exp.csv') #

df_known['smiles1'] = df_known['smiles1'].str.replace('*', 'C')
df_known['smiles2'] = df_known['smiles2'].str.replace('*', 'C')
df_known['smiles3'] = df_known['smiles3'].str.replace('*', 'C')
df_known['percentage_1'] = df_known['percentage_1']/100
df_known['percentage_2'] = df_known['percentage_2']/100
df_known['percentage_3'] = df_known['percentage_3']/100

# print('tza', all_possibilities[all_possibilities['smiles3']=='0'])
# print('tza1',df_known[df_known['smiles3']=='0'])
columns_to_compare= ['smiles1', 'percentage_1', 'smiles2', 'percentage_2', 'smiles3', 'percentage_3']
merged = all_possibilities[all_possibilities['smiles3']=='0'].merge(df_known[df_known['smiles3']=='0'][columns_to_compare], on=columns_to_compare, how='left', indicator=True)
# # # # # print(all_possibilities.columns.values)
# print('merged', merged.shape)
# merged['exclude'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)
# merged = merged.drop('_merge', axis=1) # 4. Save the Updated CSV 
# print(merged.shape)
# merged.to_csv('updated_csv2.csv', index=False)
#print(all_possibilities[all_possibilities['exclude']==1].shape)
# merged[merged['exclude']==0].reset_index(drop=True).to_csv('updated_csv2.csv', index=False)
#print(merged[merged['exclude']==1].reset_index(drop=True))
#print('index', merged[merged['exclude']==1])
#all_possibilities  = merged[merged['exclude']==0].reset_index(drop=True)


# all_possibilities.to_csv('all_possibilities.csv', index=None)


# all_possibilities = all_possibilities.drop('exclude', axis=1)
# all_possibilities = all_possibilities.drop_duplicates(subset = ['smiles1', 'percentage_1', 'smiles2', 'percentage_2', 'smiles3', 'percentage_3']).reset_index(drop=True)

all_possibilities['exclude'] = 0
all_possibilities = all_possibilities.reset_index(drop=True)



# 
print('new_shape',all_possibilities.shape)
# all_possibilities=pd.read_csv('updated_csv2.csv') 
# print('new_shape',all_possibilities.shape)
iter_run(init_file, 0)


# all_possibilities = pd.read_csv('all_possibilities.csv')
# print('new_shape',all_possibilities.shape)
# tested_candidates = pd.read_csv('all_results_loop1_new.csv')
# columns_to_compare =['smiles1', 'percentage_1', 'smiles2', 'percentage_2', 'smiles3', 'percentage_3']

# merged = all_possibilities.merge(tested_candidates[columns_to_compare], on=columns_to_compare, how='left', indicator=True)
# # # # # print(all_possibilities.columns.values)
# print(merged.shape)
# merged['exclude'] = merged['_merge'].apply(lambda x: 1 if x == 'both' else 0)
# merged = merged.drop('_merge', axis=1) # 4. Save the Updated CSV 
# print(merged.shape)
# merged.to_csv('updated_csv2.csv', index=False)
# print(all_possibilities[all_possibilities['exclude']==1].shape)
# merged[merged['exclude']==0].reset_index(drop=True).to_csv('updated_csv2.csv', index=False)
#print(merged[merged['exclude']==0].reset_index(drop=True))