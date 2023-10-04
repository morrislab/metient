U_CUTOFF = 0.05
# the higher this value, the closer to 0.0 the genetic distance is when put into e^(GENETIC_ALPHA*x)
G_IDENTICAL_CLONE_VALUE = 2.0
ORGANOTROP_ALPHA = -5.0
GENETIC_ALPHA = -5.0

import logging
logger = logging.getLogger('SGD')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s\n\r%(message)s', datefmt='%H:%M:%S')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

MIG_KEY = "migration_number"
COMIG_KEY = "comigration_number"
SEEDING_KEY = "seeding_site_number"
ORGANOTROP_KEY = "organotropism"
ENTROPY_KEY = "negative_entropy"
GEN_DIST_KEY = "genetic_distance"
DATA_FIT_KEY = "neg_log_likelihood"
REG_KEY = "regularizer"
FULL_LOSS_KEY = "loss"

# For pickle outputs
OUT_LABElING_KEY = "ancestral_labelings"
OUT_LOSSES_KEY = "losses"
OUT_IDX_LABEL_KEY = "full_tree_node_idx_to_label"
OUT_ADJ_KEY = "full_adjacency_matrix"
OUT_SITES_KEY = "ordered_anatomical_sites"
OUT_PRIMARY_KEY = "primary_site"
OUT_SUB_PRES_KEY = "subclonal_presence_matrix"

# From: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6029450/#SD8
CANCER_DRIVER_GENES = ['CDK4', 'FGFR3', 'RHOA', 'PHF6', 'TGFBR2', 'POLRMT', 'EP300', 'RET', 'RQCD1', 'JAK3', 'PIM1', 'GPS2', 'GNAS', 'ABL1', 'BRCA1', 'IRF6', 'RRAS2', 'TLR4', 'ATF7IP', 'CASP8', 'MAP2K1', 'MEN1', 'CDKN2C', 'TET2', 'CBWD3', 'EGFR', 'IDH1', 'MTOR', 'PIK3CB', 'RPS6KA3', 'PMS1', 'EEF2', 'RPL22', 'RAD21', 'PTPN11', 'NFE2L2', 'CIC', 'BCL2L11', 'SETD2', 'ZNF750', 'ERBB3', 'CUL3', 'CDKN1A', 'HLA-B', 'MSH2', 'POLQ', 'SMARCB1', 'SOX9', 'NOTCH1', 'NOTCH2', 'PMS2', 'TBL1XR1', 'HIST1H1C', 'PLCG1', 'RXRA', 'UNCX', 'ARAF', 'THRAP3', 'BRD7', 'TGIF1', 'KLF5', 'BRAF', 'RARA', 'EPHA2', 'RNF111', 'APC', 'FLT3', 'MSH3', 'RAF1', 'ZMYM2', 'MAX', 'ERCC2', 'ATXN3', 'LATS2', 'ACVR1', 'PIK3R1', 'ZFHX3', 'PRKAR1A', 'KRAS', 'NF2', 'WHSC1', 'CUL1', 'MGMT', 'POLE', 'B2M', 'TCF12', 'TAF1', 'CDK12', 'EPAS1', 'XPO1', 'KMT2A', 'FAM46D', 'CACNA1A', 'ARID2', 'ATM', 'HUWE1', 'NUP133', 'GNA13', 'FOXA1', 'GTF2I', 'JAK2', 'PDS5B', 'NUP93', 'TP53', 'MSH6', 'SPTAN1', 'ELF3', 'FBXW7', 'PLCB4', 'SPOP', 'AKT1', 'CTCF', 'KEAP1', 'KRT222', 'NIPBL', 'SMARCA4', 'KDM5C', 'CHEK2', 'IRF2', 'LZTR1', 'GATA3', 'ALK', 'PPM1D', 'TCF7L2', 'VHL', 'ESR1', 'ZBTB7B', 'DDX3X', 'AXIN2', 'AMER1', 'BCOR', 'ARID1A', 'JAK1', 'CEBPA', 'CTNNB1', 'KIF1A', 'PDGFRA', 'PTCH1', 'CTNND1', 'MGA', 'RAC1', 'RFC1', 'PIK3CA', 'CYLD', 'PTMA', 'RHOB', 'NPM1', 'DIAPH2', 'MAP3K1', 'STAG2', 'ZFP36L1', 'PTPDC1', 'PCBP1', 'SOX17', 'KANSL1', 'LATS1', 'MED12', 'WT1', 'BRCA2', 'SETBP1', 'EGR3', 'TMSB4X', 'DNMT3A', 'AJUBA', 'LEMD2', 'SMC3', 'MYH9', 'ZFP36L2', 'EZH2', 'FOXQ1', 'COL5A1', 'CDKN1B', 'MET', 'PAX5', 'KEL', 'CHD4', 'CSDE1', 'PPP2R1A', 'RIT1', 'SF1', 'CHD8', 'INPPL1', 'TXNIP', 'ACVR1B', 'SMAD4', 'RBM10', 'FGFR2', 'IL7R', 'ATRX', 'PSIP1', 'DHX9', 'RUNX1', 'FAT1', 'GNA11', 'CHD3', 'SCAF4', 'MACF1', 'NRAS', 'HLA-A', 'ASXL1', 'BTG2', 'HGF', 'KMT2C', 'MYC', 'RNF43', 'ZCCHC12', 'USP9X', 'SIN3A', 'HIST1H1E', 'NCOR1', 'ATR', 'DICER1', 'MAPK1', 'ERBB4', 'CREB3L3', 'SRSF2', 'PTPRC', 'MECOM', 'CYSLTR2', 'MYCN', 'SOS1', 'MAP2K4', 'U2AF1', 'CD70', 'KMT2D', 'RHEB', 'SMAD2', 'EIF1AX', 'AXIN1', 'KDM6A', 'RPL5', 'TSC2', 'ZBTB20', 'FOXA2', 'CDH1', 'ZNF133', 'PIK3CG', 'TSC1', 'NSD1', 'BCL2', 'STK11', 'TNFAIP3', 'CCND1', 'EEF1A1', 'TRAF3', 'BAP1', 'PBRM1', 'ARHGAP35', 'KMT2B', 'TCEB1', 'TBX3', 'DMD', 'EPHA3', 'ERBB2', 'SPTA1', 'MLH1', 'SMARCA1', 'H3F3C', 'SMC1A', 'RASA1', 'CARD11', 'CD79B', 'IDH2', 'ZC3H12A', 'ALB', 'MYD88', 'GNAQ', 'CNBD1', 'HRAS', 'SF3B1', 'FUBP1', 'RB1', 'DACH1', 'NF1', 'PIK3R2', 'GRIN2D', 'H3F3A', 'FGFR1', 'PGR', 'MAP3K4', 'ASXL2', 'GABRA6', 'CREBBP', 'PTEN', 'CDKN2A', 'CBFB', 'APOB', 'ZMYM3', 'ACVR2A', 'PTPRD', 'PPP6C', 'ARID5B', 'PLXNB2', 'IL6ST', 'MUC6', 'DAZAP1', 'KIT', 'FLNA', 'AR']
ENSEMBLE_TO_GENE_MAP = {'ENSG00000135446': 'CDK4', 'ENSG00000068078': 'FGFR3', 'ENSG00000067560': 'RHOA', 'ENSG00000156531': 'PHF6', 'ENSG00000163513': 'TGFBR2', 'ENSG00000099821': 'POLRMT', 'ENSG00000100393': 'EP300', 'ENSG00000165731': 'RET', 'ENSG00000144580': 'RQCD1', 'ENSG00000105639': 'JAK3', 'ENSG00000196365': 'PIM1', 'ENSG00000132522': 'GPS2', 'ENSG00000087460': 'GNAS', 'ENSG00000097007': 'ABL1', 'ENSG00000012048': 'BRCA1', 'ENSG00000117595': 'IRF6', 'ENSG00000133818': 'RRAS2', 'ENSG00000136869': 'TLR4', 'ENSG00000171681': 'ATF7IP', 'ENSG00000064012': 'CASP8', 'ENSG00000169032': 'MAP2K1', 'ENSG00000133895': 'MEN1', 'ENSG00000123080': 'CDKN2C', 'ENSG00000168769': 'TET2', 'ENSG00000196873': 'CBWD3', 'ENSG00000146648': 'EGFR', 'ENSG00000138413': 'IDH1', 'ENSG00000198793': 'MTOR', 'ENSG00000051382': 'PIK3CB', 'ENSG00000177189': 'RPS6KA3', 'ENSG00000064933': 'PMS1', 'ENSG00000167658': 'EEF2', 'ENSG00000116251': 'RPL22', 'ENSG00000164754': 'RAD21', 'ENSG00000179295': 'PTPN11', 'ENSG00000116044': 'NFE2L2', 'ENSG00000079432': 'CIC', 'ENSG00000153094': 'BCL2L11', 'ENSG00000181555': 'SETD2', 'ENSG00000141579': 'ZNF750', 'ENSG00000065361': 'ERBB3', 'ENSG00000036257': 'CUL3', 'ENSG00000124762': 'CDKN1A', 'ENSG00000234745': 'HLA-B', 'ENSG00000095002': 'MSH2', 'ENSG00000051341': 'POLQ', 'ENSG00000099956': 'SMARCB1', 'ENSG00000125398': 'SOX9', 'ENSG00000148400': 'NOTCH1', 'ENSG00000134250': 'NOTCH2', 'ENSG00000122512': 'PMS2', 'ENSG00000177565': 'TBL1XR1', 'ENSG00000187837': 'HIST1H1C', 'ENSG00000124181': 'PLCG1', 'ENSG00000186350': 'RXRA', 'ENSG00000164853': 'UNCX', 'ENSG00000078061': 'ARAF', 'ENSG00000054118': 'THRAP3', 'ENSG00000166164': 'BRD7', 'ENSG00000177426': 'TGIF1', 'ENSG00000102554': 'KLF5', 'ENSG00000157764': 'BRAF', 'ENSG00000131759': 'RARA', 'ENSG00000142627': 'EPHA2', 'ENSG00000157450': 'RNF111', 'ENSG00000134982': 'APC', 'ENSG00000122025': 'FLT3', 'ENSG00000113318': 'MSH3', 'ENSG00000169397': 'RAF1', 'ENSG00000121741': 'ZMYM2', 'ENSG00000125952': 'MAX', 'ENSG00000104884': 'ERCC2', 'ENSG00000066427': 'ATXN3', 'ENSG00000150457': 'LATS2', 'ENSG00000115170': 'ACVR1', 'ENSG00000145675': 'PIK3R1', 'ENSG00000140836': 'ZFHX3', 'ENSG00000108946': 'PRKAR1A', 'ENSG00000133703': 'KRAS', 'ENSG00000186575': 'NF2', 'ENSG00000109685': 'WHSC1', 'ENSG00000055130': 'CUL1', 'ENSG00000170430': 'MGMT', 'ENSG00000177084': 'POLE', 'ENSG00000166710': 'B2M', 'ENSG00000140262': 'TCF12', 'ENSG00000147133': 'TAF1', 'ENSG00000167258': 'CDK12', 'ENSG00000116016': 'EPAS1', 'ENSG00000082898': 'XPO1', 'ENSG00000118058': 'KMT2A', 'ENSG00000174016': 'FAM46D', 'ENSG00000141837': 'CACNA1A', 'ENSG00000189079': 'ARID2', 'ENSG00000149311': 'ATM', 'ENSG00000086758': 'HUWE1', 'ENSG00000069248': 'NUP133', 'ENSG00000120063': 'GNA13', 'ENSG00000129514': 'FOXA1', 'ENSG00000263001': 'GTF2I', 'ENSG00000096968': 'JAK2', 'ENSG00000083642': 'PDS5B', 'ENSG00000102900': 'NUP93', 'ENSG00000141510': 'TP53', 'ENSG00000116062': 'MSH6', 'ENSG00000197694': 'SPTAN1', 'ENSG00000163435': 'ELF3', 'ENSG00000109670': 'FBXW7', 'ENSG00000101333': 'PLCB4', 'ENSG00000121067': 'SPOP', 'ENSG00000142208': 'AKT1', 'ENSG00000102974': 'CTCF', 'ENSG00000079999': 'KEAP1', 'ENSG00000213424': 'KRT222', 'ENSG00000164190': 'NIPBL', 'ENSG00000127616': 'SMARCA4', 'ENSG00000126012': 'KDM5C', 'ENSG00000183765': 'CHEK2', 'ENSG00000168310': 'IRF2', 'ENSG00000099949': 'LZTR1', 'ENSG00000107485': 'GATA3', 'ENSG00000171094': 'ALK', 'ENSG00000170836': 'PPM1D', 'ENSG00000148737': 'TCF7L2', 'ENSG00000134086': 'VHL', 'ENSG00000091831': 'ESR1', 'ENSG00000160685': 'ZBTB7B', 'ENSG00000215301': 'DDX3X', 'ENSG00000168646': 'AXIN2', 'ENSG00000184675': 'AMER1', 'ENSG00000183337': 'BCOR', 'ENSG00000117713': 'ARID1A', 'ENSG00000162434': 'JAK1', 'ENSG00000245848': 'CEBPA', 'ENSG00000168036': 'CTNNB1', 'ENSG00000130294': 'KIF1A', 'ENSG00000134853': 'PDGFRA', 'ENSG00000185920': 'PTCH1', 'ENSG00000198561': 'CTNND1', 'ENSG00000257335': 'MGA', 'ENSG00000136238': 'RAC1', 'ENSG00000173638': 'RFC1', 'ENSG00000121879': 'PIK3CA', 'ENSG00000083799': 'CYLD', 'ENSG00000187514': 'PTMA', 'ENSG00000143878': 'RHOB', 'ENSG00000181163': 'NPM1', 'ENSG00000147202': 'DIAPH2', 'ENSG00000095015': 'MAP3K1', 'ENSG00000101972': 'STAG2', 'ENSG00000185650': 'ZFP36L1', 'ENSG00000158079': 'PTPDC1', 'ENSG00000169564': 'PCBP1', 'ENSG00000164736': 'SOX17', 'ENSG00000120071': 'KANSL1', 'ENSG00000131023': 'LATS1', 'ENSG00000184634': 'MED12', 'ENSG00000184937': 'WT1', 'ENSG00000139618': 'BRCA2', 'ENSG00000152217': 'SETBP1', 'ENSG00000179388': 'EGR3', 'ENSG00000205542': 'TMSB4X', 'ENSG00000119772': 'DNMT3A', 'ENSG00000129474': 'AJUBA', 'ENSG00000161904': 'LEMD2', 'ENSG00000108055': 'SMC3', 'ENSG00000100345': 'MYH9', 'ENSG00000152518': 'ZFP36L2', 'ENSG00000106462': 'EZH2', 'ENSG00000164379': 'FOXQ1', 'ENSG00000130635': 'COL5A1', 'ENSG00000111276': 'CDKN1B', 'ENSG00000105976': 'MET', 'ENSG00000196092': 'PAX5', 'ENSG00000197993': 'KEL', 'ENSG00000111642': 'CHD4', 'ENSG00000009307': 'CSDE1', 'ENSG00000105568': 'PPP2R1A', 'ENSG00000143622': 'RIT1', 'ENSG00000136931': 'SF1', 'ENSG00000100888': 'CHD8', 'ENSG00000165458': 'INPPL1', 'ENSG00000265972': 'TXNIP', 'ENSG00000135503': 'ACVR1B', 'ENSG00000141646': 'SMAD4', 'ENSG00000182872': 'RBM10', 'ENSG00000066468': 'FGFR2', 'ENSG00000168685': 'IL7R', 'ENSG00000085224': 'ATRX', 'ENSG00000164985': 'PSIP1', 'ENSG00000135829': 'DHX9', 'ENSG00000159216': 'RUNX1', 'ENSG00000083857': 'FAT1', 'ENSG00000088256': 'GNA11', 'ENSG00000170004': 'CHD3', 'ENSG00000156304': 'SCAF4', 'ENSG00000127603': 'MACF1', 'ENSG00000213281': 'NRAS', 'ENSG00000206503': 'HLA-A', 'ENSG00000171456': 'ASXL1', 'ENSG00000159388': 'BTG2', 'ENSG00000019991': 'HGF', 'ENSG00000055609': 'KMT2C', 'ENSG00000136997': 'MYC', 'ENSG00000108375': 'RNF43', 'ENSG00000174460': 'ZCCHC12', 'ENSG00000124486': 'USP9X', 'ENSG00000169375': 'SIN3A', 'ENSG00000168298': 'HIST1H1E', 'ENSG00000141027': 'NCOR1', 'ENSG00000169604': 'ATR', 'ENSG00000100697': 'DICER1', 'ENSG00000100030': 'MAPK1', 'ENSG00000178568': 'ERBB4', 'ENSG00000060566': 'CREB3L3', 'ENSG00000161547': 'SRSF2', 'ENSG00000081237': 'PTPRC', 'ENSG00000085276': 'MECOM', 'ENSG00000152207': 'CYSLTR2', 'ENSG00000134323': 'MYCN', 'ENSG00000115904': 'SOS1', 'ENSG00000065559': 'MAP2K4', 'ENSG00000160201': 'U2AF1', 'ENSG00000125726': 'CD70', 'ENSG00000167548': 'KMT2D', 'ENSG00000229927': 'RHEB', 'ENSG00000175387': 'SMAD2', 'ENSG00000173674': 'EIF1AX', 'ENSG00000103126': 'AXIN1', 'ENSG00000147050': 'KDM6A', 'ENSG00000122406': 'RPL5', 'ENSG00000103197': 'TSC2', 'ENSG00000181722': 'ZBTB20', 'ENSG00000125798': 'FOXA2', 'ENSG00000105325': 'CDH1', 'ENSG00000125846': 'ZNF133', 'ENSG00000105851': 'PIK3CG', 'ENSG00000165699': 'TSC1', 'ENSG00000165671': 'NSD1', 'ENSG00000171791': 'BCL2', 'ENSG00000118046': 'STK11', 'ENSG00000118503': 'TNFAIP3', 'ENSG00000110092': 'CCND1', 'ENSG00000156508': 'EEF1A1', 'ENSG00000131323': 'TRAF3', 'ENSG00000151276': 'BAP1', 'ENSG00000163939': 'PBRM1', 'ENSG00000160007': 'ARHGAP35', 'ENSG00000272333': 'KMT2B', 'ENSG00000154582': 'TCEB1', 'ENSG00000135111': 'TBX3', 'ENSG00000198947': 'DMD', 'ENSG00000044524': 'EPHA3', 'ENSG00000141736': 'ERBB2', 'ENSG00000163554': 'SPTA1', 'ENSG00000076242': 'MLH1', 'ENSG00000102038': 'SMARCA1', 'ENSG00000188375': 'H3F3C', 'ENSG00000072501': 'SMC1A', 'ENSG00000145715': 'RASA1', 'ENSG00000198286': 'CARD11', 'ENSG00000007312': 'CD79B', 'ENSG00000182054': 'IDH2', 'ENSG00000163874': 'ZC3H12A', 'ENSG00000163631': 'ALB', 'ENSG00000172936': 'MYD88', 'ENSG00000156052': 'GNAQ', 'ENSG00000176571': 'CNBD1', 'ENSG00000174775': 'HRAS', 'ENSG00000087365': 'SF3B1', 'ENSG00000162613': 'FUBP1', 'ENSG00000139687': 'RB1', 'ENSG00000276644': 'DACH1', 'ENSG00000196712': 'NF1', 'ENSG00000105647': 'PIK3R2', 'ENSG00000105464': 'GRIN2D', 'ENSG00000163041': 'H3F3A', 'ENSG00000077782': 'FGFR1', 'ENSG00000082175': 'PGR', 'ENSG00000085511': 'MAP3K4', 'ENSG00000143970': 'ASXL2', 'ENSG00000145863': 'GABRA6', 'ENSG00000005339': 'CREBBP', 'ENSG00000171862': 'PTEN', 'ENSG00000147889': 'CDKN2A', 'ENSG00000067955': 'CBFB', 'ENSG00000084674': 'APOB', 'ENSG00000147130': 'ZMYM3', 'ENSG00000121989': 'ACVR2A', 'ENSG00000153707': 'PTPRD', 'ENSG00000119414': 'PPP6C', 'ENSG00000150347': 'ARID5B', 'ENSG00000196576': 'PLXNB2', 'ENSG00000134352': 'IL6ST', 'ENSG00000184956': 'MUC6', 'ENSG00000071626': 'DAZAP1', 'ENSG00000157404': 'KIT', 'ENSG00000196924': 'FLNA', 'ENSG00000109321': 'AR'}