import pstats
import io
import sys

profile_file = "/home/ubuntu/CIFO_EXTENDED_Project/images_sp/timing_estimate/profiling_results/timing_estimate_profile.prof"
analysis_file_path = "/home/ubuntu/CIFO_EXTENDED_Project/images_sp/timing_estimate/profiling_results/profile_analysis.txt"

# Redirect stdout to capture pstats output
old_stdout = sys.stdout
sys.stdout = captured_output = io.StringIO()

results_for_file = []

try:
    stats = pstats.Stats(profile_file)
    stats.strip_dirs() # Remove path information

    results_for_file.append("Top 30 functions by cumulative time:")
    stats.sort_stats("cumulative").print_stats(30)
    results_for_file.append("\n" + "-"*80 + "\n")

    results_for_file.append("Stats for 'is_valid':")
    stats.print_stats("is_valid")
    results_for_file.append("\n" + "-"*80 + "\n")

    results_for_file.append("Stats for 'fitness':")
    stats.print_stats("fitness")
    results_for_file.append("\n" + "-"*80 + "\n")
    
    results_for_file.append("Stats for 'get_neighbors' (HC specific method in LeagueHillClimbingSolution):")
    stats.print_stats("LeagueHillClimbingSolution.*get_neighbors") # More specific regex
    results_for_file.append("\n" + "-"*80 + "\n")

    results_for_file.append("Stats for 'get_random_neighbor' (SA specific method in LeagueSASolution):")
    stats.print_stats("LeagueSASolution.*get_random_neighbor") # More specific regex
    results_for_file.append("\n" + "-"*80 + "\n")

    results_for_file.append("Stats for 'deepcopy':")
    stats.print_stats("deepcopy")
    results_for_file.append("\n" + "-"*80 + "\n")
    
    results_for_file.append("Stats for methods within solution.py:")
    stats.print_stats("solution.py")
    results_for_file.append("\n" + "-"*80 + "\n")

    results_for_file.append("Stats for methods within evolution.py:")
    stats.print_stats("evolution.py")
    results_for_file.append("\n" + "-"*80 + "\n")

finally:
    sys.stdout = old_stdout

analysis_results_str = captured_output.getvalue()
print(analysis_results_str) # This will be captured by the shell_exec observation

with open(analysis_file_path, "w") as f:
    f.write("\n".join(results_for_file)) # Write the collected lines
    f.write("\n\nFull PStats Output (as captured by print_stats directly):\n")
    f.write(analysis_results_str)

print(f"Profiling analysis also saved to {analysis_file_path}")

