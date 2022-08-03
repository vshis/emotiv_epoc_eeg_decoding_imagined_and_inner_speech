
function initialize(box)

	dofile(box:get_config("${Path_Data}") .. "/plugins/stimulation/lua-stimulator-stim-codes.lua")

	-- each stimulation sent that gets rendered by Display Cue Image box 
	-- should probably have a little period of time before the next one or the box wont be happy
	
	-- when changing timings, remember to change them in the corresponding epoching boxes too
	baseline_duration = 5  -- before the whole experiment begins. default: 5
	cue_duration = 3  -- image of phoneme. default: 5
	prepare_articulators_duration = 1  -- prepare articulators. default: 2
	thinking_duration = 3  -- blank white screen. default: 5
	--speaking_duration = 5  -- speaking. default: 5
	rest_duration = 3  -- rest. default: 5
	post_trial_duration = 1  -- before the next phoneme is displayed. default: 1
	
	sequence = {
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_0B,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_0B,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_0B,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_0B,
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_0B,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_01,
		OVTK_StimulationId_Label_0D,
		OVTK_StimulationId_Label_07,
		OVTK_StimulationId_Label_08,
		OVTK_StimulationId_Label_09,
		OVTK_StimulationId_Label_0A,
		OVTK_StimulationId_Label_0E,
		OVTK_StimulationId_Label_03,
		OVTK_StimulationId_Label_10,
		OVTK_StimulationId_Label_02,
		OVTK_StimulationId_Label_06,
		OVTK_StimulationId_Label_0F,
		OVTK_StimulationId_Label_0C,
		OVTK_StimulationId_Label_04,
		OVTK_StimulationId_Label_05,
		OVTK_StimulationId_Label_0B,													
	}

	--[[
	OVTK_StimulationId_Label_01 : p
    OVTK_StimulationId_Label_02 : t
    OVTK_StimulationId_Label_03 : k
    OVTK_StimulationId_Label_04 : f
    OVTK_StimulationId_Label_05 : s
    OVTK_StimulationId_Label_06 : sh
    OVTK_StimulationId_Label_07 : v
    OVTK_StimulationId_Label_08 : z
    OVTK_StimulationId_Label_09 : zh
    OVTK_StimulationId_Label_0A : m
    OVTK_StimulationId_Label_0B : n
    OVTK_StimulationId_Label_0C : ng
    OVTK_StimulationId_Label_0D : fleece
    OVTK_StimulationId_Label_0E : goose
    OVTK_StimulationId_Label_0F : trap
    OVTK_StimulationId_Label_10 : thought
	OVTK_StimulationId_Label_11 : gnaw
	OVTK_StimulationId_Label_12 : knew
	OVTK_StimulationId_Label_13 : pot
	OVTK_StimulationId_Label_14 : pat
	OVTK_StimulationId_Label_15 : diy
	OVTK_StimulationId_Label_16 : tiy
	OVTK_StimulationId_Label_17 : piy
	OVTK_StimulationId_Label_18 : uw
	OVTK_StimulationId_Label_19 : iy
	--]]

end

function process(box)

	local t = 0

	-- Delays before the trial sequence starts

	box:send_stimulation(1, OVTK_StimulationId_BaselineStart, t, 0)
	t = t + baseline_duration

	-- creates each trial
	for i = 1, #sequence do

		box:send_stimulation(1, OVTK_GDF_Start_Of_Trial, t, 0)
			
		--phoneme
		box:send_stimulation(1, OVTK_StimulationId_Label_00, t, 0)
		box:send_stimulation(1, sequence[i], t, 0)
		t = t + cue_duration
		
		--prepare_articulators
		box:send_stimulation(1, OVTK_GDF_Tongue_Movement, t, 0)
		t = t + prepare_articulators_duration

		--thinking
		box:send_stimulation(1, OVTK_GDF_Feedback_Continuous, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_Number_00, t, 0)
		t = t + thinking_duration/5
		box:send_stimulation(1, OVTK_StimulationId_Number_01, t, 0)
		t = t + thinking_duration/5
		box:send_stimulation(1, OVTK_StimulationId_Number_02, t, 0)
		t = t + thinking_duration/5
		box:send_stimulation(1, OVTK_StimulationId_Number_03, t, 0)
		t = t + thinking_duration/5
		box:send_stimulation(1, OVTK_StimulationId_Number_04, t, 0)
		t = t + thinking_duration/5

		--speaking
		--box:send_stimulation(1, OVTK_GDF_Tongue, t, 0)
		--t = t + speaking_duration
		
		--rest
		box:send_stimulation(1, OVTK_StimulationId_RestStart, t, 0)
		t = t + rest_duration
		
		-- end of thinking epoch and trial
		box:send_stimulation(1, OVTK_StimulationId_VisualStimulationStop, t, 0)
		box:send_stimulation(1, OVTK_StimulationId_RestStop, t, 0)
		t = t + post_trial_duration
		box:send_stimulation(1, OVTK_GDF_End_Of_Trial, t, 0)	
	end

	-- send end for completeness	
	box:send_stimulation(1, OVTK_GDF_End_Of_Session, t, 0)
	t = t + 5

	-- used to cause the acquisition scenario to stop and denote final end of file
	box:send_stimulation(1, OVTK_StimulationId_ExperimentStop, t, 0)
		
	print(t)
end
