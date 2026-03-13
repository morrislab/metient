
import pandas as pd
import pyreadr
import json
import unittest

class TestTRACERxPreprocessing(unittest.TestCase):

	def _get_sample_to_ref_var_count(self, region_sum):
		'''
		Args:
			region_sum: from TRACERx input mutTableAll.cloneInfo.20220726.rda RegionSum 
			column, e.g. R_LN1:40/168;BR_LN2:64/276;BR_LN3:47/235;SU_FLN1:96/518

		Returns:
			dictionary mapping sample name to tuple of (ref, total) counts
		'''
		d = dict()
		region_sum_values = region_sum.split(";")
		for sample_info in region_sum_values:
			items = sample_info.split(":")
			counts = items[1].split("/")
			d[items[0]] = int(counts[0]), int(counts[1])
		return d

	def _get_sample_to_cn_count(self, cn):
		'''
		Args:
			region_sum: from TRACERx input mutTableAll.cloneInfo.20220726.rda MajorCPN_SC
			or MinoRCPN_SC column, e.g. SU_T1.R1:3;SU_T1.R2:2;SU_FLN1:2;

		Returns:
			dictionary mapping sample name to tuple of copy number counts
		'''
		d = dict()
		if isinstance(cn, float):
			return d
		cn_values = cn.split(";")
		for sample_info in cn_values:
			items = sample_info.split(":")
			cn_count = int(items[1])
			d[items[0]] = cn_count
		return d

	def _test_mig_hist_pt_orchard_tsv(self, patient_id, patient_tsv_dir, true_patient_data):
		'''
		Tests tsvs used for MACHINA and Metient input and ssm/params.json files
		used for PairTree and Orchard input against ground truth data
		'''
		# Load preprocessed data
		patient_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}_SNVs.tsv"), sep="\t")
		ssm_df = pd.read_csv(os.path.join(patient_tsv_dir, f"{patient_id}.ssm"), sep="\t")
		with open(os.path.join(patient_tsv_dir, f"{patient_id}.params.json")) as f:
			params_json = json.load(f)
		json_sample_names = params_json['samples']
		#print(params_json['samples'])
		#print(ssm_df['var_reads'])
		sample_names = patient_df['sample_label'].unique()
		mut_names = patient_df['character_label'].unique()
		anatomical_site_names = patient_df['anatomical_site_label'].unique()

		self.assertEqual(len(sample_names), len(json_sample_names))
		self.assertEqual(len(patient_df), len(sample_names)*len(mut_names))

		true_patient_data = true_patient_data[true_patient_data['patient_id']==patient_id]
		for _, row in patient_df.iterrows():
			mut = (":").join(row['character_label'].split(":")[1:])
			sample = ("_").join(row['sample_label'].split("_")[1:])

			ssm_row = ssm_df[ssm_df['name'] == row['character_label']]
			json_sample_index = json_sample_names.index(row['sample_label'])

			# e.g. R_LN1:40/168;BR_LN2:64/276;BR_LN3:47/235;SU_FLN1:96/518
			true_mut_subset = true_patient_data[(true_patient_data['mutation_id']==f"{patient_id}:{mut}")]['RegionSum'].iloc[0]
			true_sample_to_counts = self._get_sample_to_ref_var_count(true_mut_subset)
			
			# Check tsvs
			self.assertEqual(true_sample_to_counts[sample][0], row['var'])
			self.assertEqual(true_sample_to_counts[sample][1], row['var']+row['ref'])

			# Check ssm
			ssm_var_count = int(ssm_row['var_reads'].item().split(",")[json_sample_index])
			ssm_total_count = int(ssm_row['total_reads'].item().split(",")[json_sample_index])
			self.assertEqual(true_sample_to_counts[sample][0], ssm_var_count)
			self.assertEqual(true_sample_to_counts[sample][1], ssm_total_count)

	def _test_conipher_tsv(self, patient_id, conipher_tsv_dir, true_patient_data, true_sample_data, true_purity_ploidy):
		'''
		Tests tsvs used for input into CONIPHER (which does clustering via pyclone
		+ tree building). See: https://github.com/McGranahanLab/CONIPHER-wrapper
		'''
		patient_df = pd.read_csv(os.path.join(conipher_tsv_dir, f"{patient_id}_conipher_SNVs.tsv"), sep="\t")
		sample_names = patient_df['SAMPLE'].unique()
		case_id = patient_df['CASE_ID'].unique()
		self.assertEqual(case_id, patient_id)

		true_patient_data = true_patient_data[true_patient_data['patient_id']==patient_id]
		true_sample_data = true_sample_data[true_sample_data['patient_id']==patient_id]

		num_expected_rows = 0
		for _, row in true_patient_data.iterrows():
			samples_w_cn = set(self._get_sample_to_cn_count(row['MajorCPN_SC']).keys()).intersection(self._get_sample_to_cn_count(row['MinorCPN_SC']).keys())
			samples_w_cn_and_mut_data = samples_w_cn.intersection(set(self._get_sample_to_ref_var_count(row['RegionSum']).keys()))
			samples_w_sample_info = set([region.replace(f"{patient_id}_", "") for region in true_sample_data['region']])
			samples_w_full_info = samples_w_cn_and_mut_data.intersection(samples_w_sample_info)
			num_expected_rows += len(samples_w_full_info)

		
		self.assertEqual(len(patient_df), num_expected_rows)

		for _, row in patient_df.iterrows():
			mut_id = f"{patient_id}:{row['CHR']}:{row['POS']}:{row['REF']}"
			sample = row['SAMPLE']
			truncated_sample = ("_").join(sample.split("_")[1:])
			true_mut_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['RegionSum'].iloc[0]
			true_major_cn_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['MajorCPN_SC'].iloc[0]
			true_minor_cn_subset = true_patient_data[true_patient_data['mutation_id']==mut_id]['MinorCPN_SC'].iloc[0]
			true_sample_to_counts = self._get_sample_to_ref_var_count(true_mut_subset)
			true_sample_to_major_cn = self._get_sample_to_cn_count(true_major_cn_subset)
			true_sample_to_minor_cn = self._get_sample_to_cn_count(true_minor_cn_subset)

			self.assertEqual(true_sample_to_counts[truncated_sample][0], row['VAR_COUNT'])
			self.assertEqual(true_sample_to_counts[truncated_sample][1], row['DEPTH'])
			self.assertEqual(row['DEPTH'] - row['VAR_COUNT'], row['REF_COUNT'])
			self.assertEqual(true_sample_to_major_cn[truncated_sample], row['COPY_NUMBER_A'])
			self.assertEqual(true_sample_to_minor_cn[truncated_sample], row['COPY_NUMBER_B'])

			# Arghhh this indexing is not working
			# true_purity_ploidy = true_purity_ploidy[true_purity_ploidy['Sample']==sample]
			# self.assertEqual(true_purity_ploidy['final.purity'].iloc[0], np.array(row['ACF']))
			# self.assertEqual(true_purity_ploidy['final.ploidy'].iloc[0], row['PLOIDY'])


	def test_tsvs(self):
		tracerx_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/tracerx_nsclc")
		patient_tsv_dir = os.path.join(tracerx_dir, "patient_data")
		conipher_tsv_dir = os.path.join(tracerx_dir, "conipher_inputs")
		# Groudn truth (original data)
		true_patient_data = pyreadr.read_r(os.path.join(tracerx_dir, 'mutTableAll.cloneInfo.20220726.rda'))['mutTableAll']
		true_sample_info_df = pd.read_csv(os.path.join(tracerx_dir,"sample_overview_original.txt"), sep="\t")
		true_purity_ploidy_df = pd.read_csv(os.path.join(tracerx_dir, "20220808_purityandploidy.txt"), sep="\t")

		patients = true_patient_data['patient_id'].unique()
		print(len(patients), "patients")
		
		for patient in patients:
			self._test_mig_hist_pt_orchard_tsv(patient, patient_tsv_dir, true_patient_data)
			self._test_conipher_tsv(patient, conipher_tsv_dir, true_patient_data, true_sample_info_df, true_purity_ploidy_df)

	# TODO: test write_pooled_tsv_from_pairtree_clusters
