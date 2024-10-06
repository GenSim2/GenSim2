# Task
This file contains all the tasks available in the project (refer to the [task folder](../gensim2/env/task)), including 7 original tasks, 21 primitive tasks, and 22 long-horizon tasks. We group tasks into sub-folders by its category and provide a dictionary of tasks with python file names and the corresponding task names as follows:
```python
# Original tasks
"open_box": OpenBox,
"close_box": CloseBox,
"open_laptop": OpenLaptop,
"close_laptop": CloseLaptop,
"turn_on_faucet": TurnOnFaucet,
"turn_off_faucet": TurnOffFaucet,
"move_mug_into_drawer": MoveMugintoDrawer,

# Primitive tasks
"open_drawer": OpenDrawer,
"push_drawer_close": PushDrawerClose,
"swing_bucket_handle": SwingBucketHandle,
"lift_bucket_upright": LiftBucketUpright,
"press_toaster_lever": PressToasterLever,
"push_toaster_forward": PushToasterForward,
"move_bag_forward": MoveBagForward,
"open_safe": OpenSafe,
"close_safe": CloseSafe,
"rotate_microwave_door": RotateMicrowaveDoor,
"close_microwave": CloseMicrowave,
"close_suitcase_lid": CloseSuitcaseLid,
"swing_suitcase_lid_open": SwingSuitcaseLidOpen,
"relocate_suitcase": RelocateSuitcase,
"swing_door_open": SwingDoorOpen,
"toggle_door_close": ToggleDoorClose,
"close_refrigerator_door": CloseRefrigeratorDoor,
"open_refrigerator_door": OpenRefrigeratorDoor,
"push_box": PushBox,
"push_oven_close": PushOvenClose,
"rotate_oven_knob": RotateOvenKnob,
"sway_bag_trap": SwayBagStrap,

# Long-horizon tasks
# Please note that long-horizon tasks can be very hard to collect demontrations due to the low success rate of sub-task chaining. 
# Also multi-task training may not work for long-horizon tasks
"insert_orange_into_microwave": InsertOrangeIntoMicrowave, 
"stash_cup_in_box": StashCupInBox, 
"fill_mug_with_water": FillMugWithWater, 
"place_cracker_box_into_refrigerator": PlaceCrackerBoxIntoRefrigerator, 
"move_cup_into_drawer": MoveCupIntoDrawer, 
"drop_spatula_in_bucket_and_lift": DropSpatulaInBucketAndLift, 
"place_softball_on_laptop": PlaceSoftballOnLaptop, 
"place_softball_into_drawer": PlaceSoftballIntoDrawer, 
"store_block_in_box": StoreBlockInBox, 
"secure_gold_in_safe": SecureGoldInSafe, 
"put_cracker_box_in_box": PutCrackerBoxInBox, 
"store_marker_in_box": StoreMarkerInBox, 
"deposit_marker_into_bucket": DepositMarkerIntoBucket, 
"place_golf_ball_into_drawer": PlaceGolfBallIntoDrawer, 
"place_cracker_box_into_drawer": PlaceCrackerBoxIntoDrawer, 
"drop_apple_into_drawer": DropAppleIntoDrawer, 
"place_lemon_into_drawer": PlaceLemonIntoDrawer, 
"store_lemon_in_refrigerator": StoreLemonInRefrigerator, 
"drop_apple_into_box": DropAppleIntoBox, 
"place_pear_into_drawer": PlacePearIntoDrawer, 
"place_golf_ball_into_box": PlaceGolfBallIntoBox, 
"store_apple_in_bucket": StoreAppleInBucket
```