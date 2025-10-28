from gym.envs.registration import register


# PSM Env
register(
    id='Hemipuncture-v0',
    entry_point='surrol.tasks.hemipuncture:Hemipuncture',
    max_episode_steps=100,
)

register(
    id='NeedleReach-v0',
    entry_point='surrol.tasks.needle_reach:NeedleReach',
    max_episode_steps=50,
)
register(
    id='NeedleReach-v1',
    entry_point='surrol.tasks.needle_reach_sphere_obstacle:NeedleReach',
    max_episode_steps=100,
)
register(
    id='NeedleReach-v2',
    entry_point='surrol.tasks.needle_reach_surface_obstacle:NeedleReach',
    max_episode_steps=100,
)
register(
    id='NeedleReach-v3',
    entry_point='surrol.tasks.needle_reach_plate_obstacle:NeedleReach',
    max_episode_steps=100,
)
register(
    id='NeedleReach-v4',
    entry_point='surrol.tasks.needle_reach_half_sphere_obstacle:NeedleReach',
    max_episode_steps=100,
)
register(
    id='NeedleReach-v5',
    entry_point='surrol.tasks.needle_reach_cylinder_obstacle:NeedleReach',
    max_episode_steps=100,
)
register(
    id='NeedleReach-v6',
    entry_point='surrol.tasks.needle_reach_complex_cylinder_obstacle:NeedleReach',
    max_episode_steps=100,
)

register(
    id='Reach-v0',
    entry_point='surrol.tasks.reach_hemisphere:Reach',
    max_episode_steps=100,
)

register(
    id='GauzeRetrieve-v0',
    entry_point='surrol.tasks.gauze_retrieve:GauzeRetrieve',
    max_episode_steps=50,
)
register(
    id='GauzeRetrieve-v1',
    entry_point='surrol.tasks.gauze_retrieve_liver:GauzeRetrieve',
    max_episode_steps=100,
)
register(
    id='GauzeRetrieve-v2',
    entry_point='surrol.tasks.gauze_retrieve_sphere:GauzeRetrieve',
    max_episode_steps=100,
)
# register(
#     id='GauzeRetrieve-v1',
#     entry_point='surrol.tasks.gauze_retrieve_sphere_obstacle:GauzeRetrieve',
#     max_episode_steps=100,
# )
# register(
#     id='GauzeRetrieve-v2',
#     entry_point='surrol.tasks.gauze_retrieve_surface_obstacle:GauzeRetrieve',
#     max_episode_steps=100,
# )
register(
    id='GauzeRetrieve-v3',
    entry_point='surrol.tasks.gauze_retrieve_plate_obstacle:GauzeRetrieve',
    max_episode_steps=100,
)
register(
    id='GauzeRetrieve-v4',
    entry_point='surrol.tasks.gauze_retrieve_half_sphere_obstacle:GauzeRetrieve',
    max_episode_steps=100,
)
register(
    id='GauzeRetrieve-v5',
    entry_point='surrol.tasks.gauze_retrieve_cylinder_obstacle:GauzeRetrieve',
    max_episode_steps=100,
)
register(
    id='GauzeRetrieve-v6',
    entry_point='surrol.tasks.gauze_retrieve_complex_cylinder_obstacle:GauzeRetrieve',
    max_episode_steps=100,
)

register(
    id='NeedlePick-v0',
    entry_point='surrol.tasks.needle_pick:NeedlePick',
    max_episode_steps=100,
)
register(
    id='NeedlePick-v1',
    entry_point='surrol.tasks.needle_pick_liver:NeedlePick',
    max_episode_steps=100,
)
register(
    id='NeedlePick-v2',
    entry_point='surrol.tasks.needle_pick_sphere:NeedlePick',
    max_episode_steps=100,
)
register(
    id='NeedlePick-v3',
    entry_point='surrol.tasks.needle_pick_hemisphere:NeedlePick',
    max_episode_steps=100,
)
# register(
#     id='NeedlePick-v1',
#     entry_point='surrol.tasks.needle_pick_sphere_obstacle:NeedlePick',
#     max_episode_steps=100,
# )
# register(
#     id='NeedlePick-v2',
#     entry_point='surrol.tasks.needle_pick_surface_obstacle:NeedlePick',
#     max_episode_steps=100,
# )
# register(
#     id='NeedlePick-v3',
#     entry_point='surrol.tasks.needle_pick_plate_obstacle:NeedlePick',
#     max_episode_steps=100,
# )
register(
    id='NeedlePick-v4',
    entry_point='surrol.tasks.needle_pick_half_sphere_obstacle:NeedlePick',
    max_episode_steps=100,
)
register(
    id='NeedlePick-v5',
    entry_point='surrol.tasks.needle_pick_cylinder_obstacle:NeedlePick',
    max_episode_steps=100,
)
register(
    id='NeedlePick-v6',
    entry_point='surrol.tasks.needle_pick_complex_cylinder_obstacle:NeedlePick',
    max_episode_steps=100,
)

register(
    id='PegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer:PegTransfer',
    max_episode_steps=50,
)
register(
    id='PegTransfer-v1',
    entry_point='surrol.tasks.peg_transfer_sphere_obstacle:PegTransfer',
    max_episode_steps=100,
)
register(
    id='PegTransfer-v2',
    entry_point='surrol.tasks.peg_transfer_surface_obstacle:PegTransfer',
    max_episode_steps=100,
)
register(
    id='PegTransfer-v3',
    entry_point='surrol.tasks.peg_transfer_plate_obstacle:PegTransfer',
    max_episode_steps=100,
)
register(
    id='PegTransfer-v4',
    entry_point='surrol.tasks.peg_transfer_half_sphere_obstacle:PegTransfer',
    max_episode_steps=100,
)
register(
    id='PegTransfer-v5',
    entry_point='surrol.tasks.peg_transfer_cylinder_obstacle:PegTransfer',
    max_episode_steps=100,
)
register(
    id='PegTransfer-v6',
    entry_point='surrol.tasks.peg_transfer_complex_cylinder_obstacle:PegTransfer',
    max_episode_steps=100,
)

# Bimanual PSM Env
register(
    id='NeedleRegrasp-v0',
    entry_point='surrol.tasks.needle_regrasp_bimanual:NeedleRegrasp',
    max_episode_steps=50,
)

register(
    id='BiPegTransfer-v0',
    entry_point='surrol.tasks.peg_transfer_bimanual:BiPegTransfer',
    max_episode_steps=50,
)

# ECM Env
register(
    id='ECMReach-v0',
    entry_point='surrol.tasks.ecm_reach:ECMReach',
    max_episode_steps=50,
)

register(
    id='MisOrient-v0',
    entry_point='surrol.tasks.ecm_misorient:MisOrient',
    max_episode_steps=50,
)

register(
    id='StaticTrack-v0',
    entry_point='surrol.tasks.ecm_static_track:StaticTrack',
    max_episode_steps=50,
)

register(
    id='ActiveTrack-v0',
    entry_point='surrol.tasks.ecm_active_track:ActiveTrack',
    max_episode_steps=500,
)
