from enum import Enum
import numpy as np 
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy import integrate
from rail.estimation.algos.naive_stack import NaiveStackSummarizer
from rail.estimation.algos.true_nz import TrueNZHistogrammer
from rail.core.stage import RailStage
import scipy
import tables_io
from scipy.special import erf
import h5py
import pandas as pd
import qp
from scipy.interpolate import UnivariateSpline
from rail.evaluation.dist_to_dist_evaluator import DistToDistEvaluator


from rail.core.data import (
    QPHandle,
    TableHandle,
    Hdf5Handle,
)

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


# Define the redshift bins and corresponding IDs
redshift_bins = {
    1: (0.2, 0.4),
    2: (0.4, 0.6),
    3: (0.6, 0.8),
    4: (0.8, 1.0),
    5: (1.0, 1.2)
}

# Function to filter and create HDF5 files for each redshift bin
def create_filtered_hdf5_files(mask, path, zphot):
    # Open the original HDF5 file in read-write mode
    with h5py.File(f'{path}/roman_rubin_y1_a_test_10sig.hdf5', 'r+') as old_file:
        # Navigate to the 'photometry' group
        if 'photometry' in old_file:
            old_photometry_group = old_file['photometry']
            if 'zphot' not in old_photometry_group:
                old_photometry_group.create_dataset('zphot', data=zphot)

            columns_to_keep = [
                "mag_err_g_lsst", "mag_err_i_lsst", "mag_err_r_lsst", 
                "mag_err_u_lsst", "mag_err_y_lsst", "mag_err_z_lsst",
                "mag_g_lsst", "mag_i_lsst", "mag_r_lsst", 
                "mag_u_lsst", "mag_y_lsst", "mag_z_lsst", 
                "redshift", "galaxy_id", 'zphot'
            ]

            # Create a filtered dataset with the mask applied
            filtered_photometry = filter_photometry(old_photometry_group, columns_to_keep, mask)

            # Create HDF5 files for each redshift bin
            for bin_i in range(1, 6):
                create_bin_hdf5_file(filtered_photometry, bin_i)

# Function to filter photometry based on the specified columns and mask
def filter_photometry(old_photometry_group, columns_to_keep, mask):
    filtered_photometry = {}
    for col in columns_to_keep:
        if col in old_photometry_group:
            filtered_photometry[col] = old_photometry_group[col][:][mask]
    return filtered_photometry

# Function to create an HDF5 file for a specific redshift bin
def create_bin_hdf5_file(filtered_photometry, bin_i):
    with h5py.File(f'roman_rubin_test_binning_{bin_i}_lens.hdf5', 'w') as new_file:
        photometry_group = new_file.create_group('photometry')

        # Get redshift data
        redshift_data = filtered_photometry['zphot'][:]

        # Find indices within the redshift range of the bin
        z_min, z_max = redshift_bins[bin_i]
        bin_indices = np.where((redshift_data >= z_min) & (redshift_data < z_max))[0]

        # Filter and copy data to the new HDF5 file
        for column in filtered_photometry:
            data = filtered_photometry[column]
            filtered_data = data[bin_indices]  # Keep only rows for the current bin
            
            # Rename 'galaxy_id' to 'id' when copying
            dataset_name = "id" if column == "galaxy_id" else column
            photometry_group.create_dataset(dataset_name, data=filtered_data)

        # Add a new 'class_id' column with the bin ID
        class_id_data = np.full(len(bin_indices), bin_i)
        photometry_group.create_dataset("class_id", data=class_id_data)

# Function to assign class IDs based on redshift values
def assign_class_id(z):
    if 0.2 <= z < 0.4:
        return 1
    elif 0.4 <= z < 0.6:
        return 2
    elif 0.6 <= z < 0.8:
        return 3
    elif 0.8 <= z < 1.0:
        return 4
    elif 1.0 <= z < 1.2:
        return 5
    else:
        return 0

# Function to create output bin files based on class IDs
def create_output_bin_file(bin_num, row_indices, class_ids):
    output_file = f'output_tomo_binned_bin{bin_num}.hdf5'
    bin_class_ids = np.where(class_ids == bin_num, bin_num, 0)

    with h5py.File(output_file, 'w') as outfile:
        outfile.create_dataset('row_index', data=row_indices)
        outfile.create_dataset('class_id', data=bin_class_ids)

    #print(f"HDF5 file '{output_file}' created successfully!")

# Function to process each bin and create the corresponding files
def process_bins():
    for bin_num in range(1, 6):
        input_file = f'roman_rubin_test_binning_{bin_num}_lens.hdf5'
        
        with h5py.File(input_file, 'r') as infile:
            redshifts = infile['photometry/zphot'][:]
            row_indices = np.arange(len(redshifts))
            class_ids = np.array([assign_class_id(z) for z in redshifts])

        # Create the output bin file
        create_output_bin_file(bin_num, row_indices, class_ids)

# Function to create histograms for tomographic bins
def create_histograms():
    for i in range(5):
        true_nz_file = f'roman_rubin_test_binning_{i + 1}_lens.hdf5'
        true_nz = DS.read_file('true_nz', path=true_nz_file, handle_class=TableHandle)

        nz_hist = TrueNZHistogrammer.make_stage(
            name=f'true_nz_lens_{i + 1}',
            hdf5_groupname='photometry',
            redshift_col='redshift',
            zmin=0.0,
            zmax=3.0,
            nzbins=301
        )

        
        tomo_file = f"output_tomo_binned_bin{i + 1}.hdf5"
        tomo_bins = DS.read_file('tomo_bins', path=tomo_file, handle_class=TableHandle)
        
        out_hist = nz_hist.histogram(true_nz, tomo_bins)
        #print(f"Histogram for bin {i + 1} created successfully.")

def plot_nz_from_bins(num_bins, param, sizes):
    colors = ['#4d4d4d', '#08306b', '#6baed6', '#ffcc00', '#ffb347']
    plt.figure(figsize=(14, 10))  # Create a new figure for plotting
    
    lens_srd = pd.read_csv('/global/u1/i/iago/lens_SRD', sep=' ', index_col=False).T
    bins = [float(x) for x in np.array(lens_srd.index)]
    bins = np.round(np.array(bins), 4)
    center_bins = [0.3,0.5,0.7,0.9,1.1]

    for i, bin_num in enumerate(range(1, num_bins + 1)):
        # Read the data from the HDF5 file
        input_file_true = qp.read(f'true_NZ_true_nz_lens_{bin_num}.hdf5')
        input_file_phot = DS.read_file(
            'pdfs_data', QPHandle, f'single_NZ_naive_stack_lens_phot_bin{bin_num-1}.hdf5'
        )

        y_true = input_file_true.objdata()['pdfs'][0]
        x_true = input_file_true.metadata()['bins'][0]

        y_phot = input_file_phot().build_tables()['data']['yvals'][0]
        x_phot = input_file_phot().build_tables()['meta']['xvals'][0]

        
        # Smoothing and normalization for true curve
        cs_true = UnivariateSpline(x_true[:-1], y_true)
        cs_true.set_smoothing_factor(1)  
        smoothed_y_true = cs_true(x_true[:-1])
        area_true = np.trapz(smoothed_y_true, x_true[:-1])
        y_true_normalized = smoothed_y_true / area_true  

        # Normalization for photometric curve
        area_phot = np.trapz(y_phot, x_phot)
        y_phot_normalized = y_phot / area_phot  
        
        
        
        ### metrics ###
        mean_true = np.trapz(y_true_normalized*x_true[:-1],x_true[:-1])
        sigma_true = np.sqrt(np.trapz(y_true_normalized*(x_true[:-1]-mean_true)**2,x_true[:-1]))
        
        mean_phot = np.trapz(y_phot_normalized*x_phot,x_phot)
        sigma_phot = np.sqrt(np.trapz(y_phot_normalized*(x_phot-mean_phot)**2,x_phot))
        
        mean_bias = mean_true-mean_phot
        sigma_bias = sigma_true-sigma_phot
        
        

        # Plot the photometric and true curves
        plt.plot(x_phot, y_phot_normalized, color=colors[i], linewidth=2)
        plt.plot(x_true[:-1], y_true_normalized, color=colors[i], linestyle='--', linewidth=2)
        #plt.plot(bins,lens_srd[i],color='red',linewidth=2)

        # Calculate the midpoint and half-height for each bin
        bin_center = center_bins[i]
        half_height = 2.5

        # Plot the metrics in the center of each bin at half-height
        metrics_text = (
            r"$\bar{z}$: "f"{np.round(mean_true,3)}\n"
            r"$\sigma$: " f"{np.round(sigma_true,3)}\n"
            r"$\Delta\bar{z}$: " f"{np.round(mean_bias,3)}\n"
            r"$\Delta\sigma$: " f"{np.round(sigma_bias,3)}\n"
            r"$\frac{N}{arcmin^2}$:" f"{np.round(sizes[i]/314937.53,3)}" #value from notebook 03 in arcmin²
        )

        plt.text(bin_center, half_height, metrics_text, 
                 ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
                 facecolor='white', alpha=0.7))
        
    plt.axvspan(0.2, 0.4, color=colors[0], alpha=0.3)  # Primeiro intervalo
    plt.axvspan(0.4, 0.6, color=colors[1], alpha=0.3)  # Segundo intervalo
    plt.axvspan(0.6, 0.8, color=colors[2], alpha=0.3)  # Terceiro intervalo
    plt.axvspan(0.8, 1.0, color=colors[3], alpha=0.3)  # Quarto intervalo
    plt.axvspan(1.0, 1.2, color=colors[4], alpha=0.3)  # Quinto intervalo
      
    
    plt.plot([], [], label='True Bin', linewidth=2, ls='--', color='black')
    plt.plot([], [], label='Phot Bin', linewidth=2, color='black')
    #plt.plot([],[], label=f'LSST DESC SRD Y1',linewidth=2,color='red')
    
    plt.legend(fontsize=14, loc='upper right', frameon=True)
    
    # Customizing the plot
    plt.xlabel('Redshift (z)', fontsize=18)
    plt.ylabel('N(z)', fontsize=18)
    plt.title(f'N(z) Distribution for lens sample for the true and observed redshifts {param}',fontsize=20)
    plt.legend(fontsize=14,loc=1)
    plt.ylim(0, 7)  # Adjusted for normalized values
    plt.xlim(0, 1.3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()



def metris_sample(num_bins, param, sizes):
    colors = ['#4d4d4d', '#08306b', '#6baed6', '#ffcc00', '#ffb347']
    plt.figure(figsize=(14, 10))  # Create a new figure for plotting
    
    lens_srd = pd.read_csv('/global/u1/i/iago/lens_SRD', sep=' ', index_col=False).T
    bins = [float(x) for x in np.array(lens_srd.index)]
    bins = np.round(np.array(bins), 4)
    center_bins = [0.3,0.5,0.7,0.9,1.1]

    mean_bias = []
    sigma_bias = []
    
    sigma_true_list = []
    
    for i, bin_num in enumerate(range(1, num_bins + 1)):
        # Read the data from the HDF5 file
        input_file_true = qp.read(f'true_NZ_true_nz_lens_{bin_num}.hdf5')
        input_file_phot = DS.read_file(
            'pdfs_data', QPHandle, f'single_NZ_naive_stack_lens_phot_bin{bin_num-1}.hdf5'
        )

        y_true = input_file_true.objdata()['pdfs'][0]
        x_true = input_file_true.metadata()['bins'][0]

        y_phot = input_file_phot().build_tables()['data']['yvals'][0]
        x_phot = input_file_phot().build_tables()['meta']['xvals'][0]

        
        # Smoothing and normalization for true curve
        cs_true = UnivariateSpline(x_true[:-1], y_true)
        cs_true.set_smoothing_factor(1)  
        smoothed_y_true = cs_true(x_true[:-1])
        area_true = np.trapz(smoothed_y_true, x_true[:-1])
        y_true_normalized = smoothed_y_true / area_true  

        # Normalization for photometric curve
        area_phot = np.trapz(y_phot, x_phot)
        y_phot_normalized = y_phot / area_phot  
        
        
        
        ### metrics ###
        mean_true = np.trapz(y_true_normalized*x_true[:-1],x_true[:-1])
        sigma_true = np.sqrt(np.trapz(y_true_normalized*(x_true[:-1]-mean_true)**2,x_true[:-1]))
        
        sigma_true_list.append(sigma_true)
        
        mean_phot = np.trapz(y_phot_normalized*x_phot,x_phot)
        sigma_phot = np.sqrt(np.trapz(y_phot_normalized*(x_phot-mean_phot)**2,x_phot))
        
        mean_bias.append(mean_true-mean_phot)
        sigma_bias.append(sigma_true-sigma_phot)
        
    return mean_bias, sigma_bias, np.round(np.array(sizes)/314937.53,3), sigma_true_list
