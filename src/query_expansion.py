from typing import List

class QueryExpander:
    def __init__(self):
        pass

    def expand_query(self, original_query: str, num_expansions: int = 3) -> List[str]:
        """
        Expand the original query into multiple related queries
        
        Parameters:
        - original_query: the original text query
        - num_expansions: number of additional queries to generate
        
        Returns:
        - expanded_queries: list of queries including the original
        """
        expanded_queries = [original_query]  # Always include the original query
        
        # Method 1: Rule-based expansion with musical terminology
        if "piano" in original_query.lower():
            expanded_queries.extend([
                original_query.replace("piano", "keyboard"),
                f"gentle {original_query}",
                f"{original_query} with soft dynamics"
            ])
        elif "guitar" in original_query.lower():
            expanded_queries.extend([
                original_query.replace("guitar", "acoustic strings"),
                f"melodic {original_query}",
                f"{original_query} riff"
            ])
        elif "vocal" in original_query.lower() or "singing" in original_query.lower():
            expanded_queries.extend([
                original_query.replace("vocal", "voice").replace("singing", "voice"),
                f"human {original_query}",
                f"{original_query} with lyrics"
            ])
            
        # Method 2: Emotion/mood-based expansion
        emotion_map = {
            "sad": ["melancholic", "somber", "emotional", "downbeat"],
            "happy": ["upbeat", "cheerful", "bright", "joyful"],
            "angry": ["intense", "aggressive", "heavy", "powerful"],
            "calm": ["peaceful", "ambient", "relaxing", "gentle"]
        }
        
        for emotion, alternatives in emotion_map.items():
            if emotion in original_query.lower():
                for alt in alternatives[:2]:  # Take just 2 alternatives
                    expanded_queries.append(original_query.replace(emotion, alt))
                
        # Fallback if we don't have enough expansions
        if len(expanded_queries) < num_expansions + 1:
            # Add some generic expansions
            generic_expansions = [
                f"{original_query} section",
                f"music with {original_query}",
                f"audio containing {original_query}",
                f"sound of {original_query}"
            ]
            expanded_queries.extend(generic_expansions)
            
        # Return original plus up to num_expansions additional queries
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
                
        return unique_queries[:num_expansions+1]