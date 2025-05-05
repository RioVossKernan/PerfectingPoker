from collections import Counter
import random

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['H', 'D', 'C', 'S']
RANK_ORDER = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
              '7': 7, '8': 8, '9': 9, '10': 10,
              'J': 11, 'Q': 12, 'K': 13, 'A': 14}

class PokerCard:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    def __repr__(self):
        return f"{self.rank}{self.suit}"

def full_deck():
    return [PokerCard(rank, suit) for rank in RANKS for suit in SUITS]

def get_rank_counts(PokerCards):
    return Counter(PokerCard.rank for PokerCard in PokerCards)

def get_suit_counts(PokerCards):
    return Counter(PokerCard.suit for PokerCard in PokerCards)

def has_flush(PokerCards, n=5):
    suit_counts = get_suit_counts(PokerCards)
    return any(count >= n for count in suit_counts.values())

def has_straight(PokerCards, n=5):
    ranks = sorted(set(RANK_ORDER[PokerCard.rank] for PokerCard in PokerCards))
    if 14 in ranks:
        ranks.insert(0, 1)
    for i in range(len(ranks) - n + 1):
        window = ranks[i:i+n]
        if all(b == a + 1 for a, b in zip(window, window[1:])):
            return True
    return False

def has_straight_flush(PokerCards, n=5):
    suits = {}
    for PokerCard in PokerCards:
        if PokerCard.suit not in suits:
            suits[PokerCard.suit] = []
        suits[PokerCard.suit].append(PokerCard)
    
    for suit, suited_PokerCards in suits.items():
        if len(suited_PokerCards) >= n and has_straight(suited_PokerCards, n):
            return True
    return False

def has_royal_flush(PokerCards):
    """Simple check for royal flush: A-K-Q-J-10 of the same suit"""
    suit_counts = get_suit_counts(PokerCards)
    flush_suit = None
    for suit, count in suit_counts.items():
        if count >= 5:
            flush_suit = suit
            break
    
    if not flush_suit:
        return False
    
    royal_ranks = {'A', 'K', 'Q', 'J', '10'}
    suit_PokerCards = [PokerCard for PokerCard in PokerCards if PokerCard.suit == flush_suit]
    suit_ranks = {PokerCard.rank for PokerCard in suit_PokerCards}
    
    return royal_ranks.issubset(suit_ranks)

def get_hand_stats(PokerCards):
    rank_counter = get_rank_counts(PokerCards)
    counts = list(rank_counter.values())
    
    return {
        'royal_flush': has_royal_flush(PokerCards),
        'straight_flush': has_straight_flush(PokerCards),
        'four_of_a_kind': 4 in counts,
        'full_house': (3 in counts and 2 in counts) or counts.count(3) >= 2,
        'flush': has_flush(PokerCards),
        'straight': has_straight(PokerCards),
        'three_of_a_kind': 3 in counts and counts.count(2) < 1,  
        'two_pair': counts.count(2) >= 2,
        'one_pair': counts.count(2) == 1 and 3 not in counts,  
        'high_PokerCard': max(counts) == 1  
    }

def estimate_hand_probabilities(current_hand, total_PokerCards=7, simulations=10000):
    results = Counter()
    current_PokerCards_set = {(c.rank, c.suit) for c in current_hand}
    deck = [PokerCard for PokerCard in full_deck() if (PokerCard.rank, PokerCard.suit) not in current_PokerCards_set]
    
    for _ in range(simulations):
        draw = random.sample(deck, total_PokerCards - len(current_hand))
        full_hand = current_hand + draw
        stats = get_hand_stats(full_hand)
        
        hand_types = ['royal_flush', 'straight_flush', 'four_of_a_kind', 
                     'full_house', 'flush', 'straight', 'three_of_a_kind', 
                     'two_pair', 'one_pair', 'high_PokerCard']
        
        for hand_type in hand_types:
            if stats[hand_type]:
                results[hand_type] += 1
                break
    
    probabilities = {hand: results[hand] / simulations for hand in results}
    return probabilities


def convert_detected_cards_to_poker_cards(cards):
    """Convert detected OpenCV cards to PokerCard objects"""
    poker_cards = []
    for card in cards:
        if card.best_rank_match != "Unknown" and card.best_suit_match != "Unknown":
            rank = card.best_rank_match
            suit_map = {"Spades": "S", "Diamonds": "D", "Clubs": "C", "Hearts": "H"}
            suit = suit_map.get(card.best_suit_match, "")
            if rank and suit:
                poker_cards.append(PokerCard(rank, suit))
    return poker_cards
