cd ../../Cosmo_Inference/
pars_file="/Users/JoeyA/Research/SPHEREx/spherex_emu/configs/get_covariance.yaml"
output_dir="/Users/JoeyA/Research/SPHEREx/spherex_emu/data/cov_2_sample_2_redshift"

# get the power spectrum
python -m lss_theory.scripts.get_data_vector $pars_file

# get the (Gaussian) covariance matrix
python -m lss_theory.scripts.get_fisher $pars_file

# reshape the covariance matrix to a more convenient shape
python -m src.lss_theory.lss_theory.covariance.reshape_multipole_covariance \
    data/cov_2_sample_2_redshift/covariance/cov.npy data/cov_2_sample_2_redshift/powerspec/redshift_bins_2_samples_2_galaxy_ps.npy

cp -r data/cov_2_sample_2_redshift/covariance/ $output_dir
cp -r data/cov_2_sample_2_redshift/powerspec/ $output_dir
