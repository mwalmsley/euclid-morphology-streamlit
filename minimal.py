import pandas as pd

if __name__ == '__main__':

    cols = [
        'right_ascension',
        'declination',
        'id_str',
        'segmentation_area',
        'mag_segmentation',
        'smooth-or-featured_smooth_fraction',
        'smooth-or-featured_featured-or-disk_fraction',
        'smooth-or-featured_problem_fraction',
        'disk-edge-on_yes_fraction',
        'disk-edge-on_no_fraction',
        'has-spiral-arms_yes_fraction',
        'has-spiral-arms_no_fraction',
        'bar_strong_fraction',
        'bar_weak_fraction',
        'bar_no_fraction',
        'bulge-size_dominant_fraction',
        'bulge-size_large_fraction',
        'bulge-size_moderate_fraction',
        'bulge-size_small_fraction',
        'bulge-size_none_fraction',
        'how-rounded_round_fraction',
        'how-rounded_in-between_fraction',
        'how-rounded_cigar-shaped_fraction',
        'edge-on-bulge_boxy_fraction',
        'edge-on-bulge_none_fraction',
        'edge-on-bulge_rounded_fraction',
        'spiral-winding_tight_fraction',
        'spiral-winding_medium_fraction',
        'spiral-winding_loose_fraction',
        'spiral-arm-count_1_fraction',
        'spiral-arm-count_2_fraction',
        'spiral-arm-count_3_fraction',
        'spiral-arm-count_4_fraction',
        'spiral-arm-count_more-than-4_fraction',
        'spiral-arm-count_cant-tell_fraction',
        'merging_none_fraction',
        'merging_minor-disturbance_fraction',
        'merging_major-disturbance_fraction',
        'merging_merger_fraction',
        # 'clumps_yes_fraction',
        # 'clumps_no_fraction',
        'problem_star_fraction',
        'problem_artifact_fraction',
        'problem_zoom_fraction',
        'artifact_satellite_fraction',
        'artifact_scattered_fraction',
        'artifact_diffraction_fraction',
        'artifact_ray_fraction',
        'artifact_saturation_fraction',
        'artifact_other_fraction',
        'artifact_ghost_fraction',
        'cutout_width_arcsec',
        'warning_galaxy_fails_training_cuts'
    ]

    # df = pd.read_parquet('/media/walml/alpha/euclid/gz_v5_q1/morphology_catalogue.parquet')
    df = pd.read_parquet('morphology_catalogue.parquet')
    # print('\n'.join(df.columns.values))
    df = df[cols]
    print(len(df))
    # df.to_parquet('/media/walml/alpha/euclid/gz_v5_q1/morphology_catalogue_minimal.parquet', index=False) 
    df.to_parquet('morphology_catalogue_minimal.parquet', index=False)    
    # print(df['spiral-arm-count_1_fraction'].isna().mean())
    # print(df['id_str'])
    
    import numpy as np
    print(np.percentile(df['cutout_width_arcsec'], 90))

    def get_field(declination):
        if declination > 40:
            return 'EDFN'
        elif declination > -20:
            return 'TODO'  # not on esa sky
        elif declination < -40:
            return 'EDFF'
        else:
            return 'EDFS'

    print(df.columns.values)
    from matplotlib import pyplot as plt
    df['tile_index'] = df['id_str'].apply(lambda x: x.split('_')[2])
    dfs = df.sample(10000)
    dfs['field'] = dfs['declination'].apply(get_field)
    plt.scatter(dfs['right_ascension'], dfs['declination'], c=dfs['field'].apply(lambda x: {'EDFN': 'r', 'EDFF': 'b', 'EDFS': 'g', 'TODO': 'k'}[x]))
    # plt.hist(df.sample(10000)['tile_index'], bins=100)
    plt.show()
