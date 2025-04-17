
cd experiment_2_and_3_combined/
for bad_name in *\.round1
do
    echo $bad_name
    mv $bad_name ${bad_name/.round1/_round1}
done
grep -R "total_train" > /home/yadunand/sc_2025_agents/collated_combined_1.0.log
