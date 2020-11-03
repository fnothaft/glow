# Copyright 2020 The Glow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glow.functions import multiply_array_by_scalar, aggregate_array_elemental_sum

import pyspark.sql.functions as F

def load_pgs(spark,
             path):
  '''
  Helper function to load in the polygenic score file.
  
  Args:
      path: Path to the polygenic score file.
  
  Returns:
      A dataframe containing the per-variant contributions to the polygenic score.
  '''
  
  return spark.read.csv(path, sep="\t", comment="#", header=True, inferSchema=True)\
    .withColumnRenamed("rsID", "name")

def calculate_pgs(genotype_df, pgs_df, attach_sample_ids=False, sample_id_df=None):
  '''
  Basic method to calculate polygenic scores given a model.
  
  Ignores but does not check for:
   - Recessive flag
   - Haplotype/diplotypes
   - Interactions
   - Inclusion criteria
   
  Args:
      genotype_df: Dataframe of biallelic genotypes, with genotype states computed.
      pgs_df: Dataframe with polygenic score model.
      attach_sample_ids: If true, returns a dataframe with one row per sample, mapping
        the predicted score to the sample ID. If false (default), returns a dataframe with
        one row containing an array with all the numeric scores.
      sample_id_df: Ignored if attach_sample_ids is false. If attach_sample_ids is true and
        this is given, this should be a dataframe with a single row containing all the sample
        IDs. If not given, then the sample IDs are collected from the first row of genotype_df.
  '''
  
  # join polygenic scores with the genotypes, using rsID
  # check for allele swaps
  named_snps = genotype_df.join(pgs_df, ["name"])\
                          .withColumn('effect',
                                      F.when(F.col('referenceAllele') == F.col('reference_allele') &&
                                             F.col('alternateAlleles').getItem(0) == F.col('effect_allele'), F.col('effect_weight'))\
                                      .when(F.col('referenceAllele') == F.col('effect_allele') &&
                                             F.col('alternateAlleles').getItem(0) == F.col('reference_allele'), -F.col('effect_weight'))\
                                      .otherwise(F.lit(0e1)))
                                                             
  
  # multiply the effect weight assigned to each variant against the genotype state of each genotype
  # this produces a contribution weight vector per scored genomic variant
  snp_effects = named_snps.withColumn('contribution',
                                      multiply_array_by_scalar(F.col('effect_weight'),
                                                               F.col('genotypeStates')))
  
  # aggregate scores across all variants using aggregate_by_index function from glow
  aggregated_snp_effects = snp_effects.agg(aggregate_array_elemental_sum(F.col('contribution'))
  
  # attach sample ids, if requested, else return
  if attach_sample_ids:
    if sample_id_df is None:
      sample_id_df = genotype_states.select(F.expr("transform(genotypes, g -> g.sampleId) as sampleId")).limit(1)
      
    return aggregated_snp_effects.crossJoin(sample_id_df)\
      .select(F.expr("arrays_zip(sampleId, contribution) as contribution"))
  
  else:
    return aggregated_snp_effects
