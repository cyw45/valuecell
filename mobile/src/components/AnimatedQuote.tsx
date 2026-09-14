import { useEffect, useRef, useState } from "react";
import {
  Animated,
  Easing,
  StyleSheet,
  Text,
  type StyleProp,
  type TextStyle,
} from "react-native";
import { useTheme } from "../theme-context";

/**
 * Live money/number readout. Mirrors the Web dashboard's animated quote: the
 * value counts up to the new fact over 650ms and plays a short scale + glow
 * pulse so a refresh is noticeable without the layout jumping. Missing facts
 * render an em dash and never animate towards a fabricated zero.
 */
export type AnimatedQuoteTone = "default" | "positive" | "negative";

export type AnimatedQuoteProps = {
  value: number | null | undefined;
  style?: StyleProp<TextStyle>;
  tone?: AnimatedQuoteTone;
  signed?: boolean;
  fractionDigits?: number;
  suffix?: string;
  durationMs?: number;
};

const COUNT_UP_MS = 650;
const PULSE_UP_MS = 200;
const PULSE_DOWN_MS = 520;

function formatNumber(value: number, fractionDigits: number, signed: boolean): string {
  const formatted = value.toLocaleString(undefined, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  });
  return `${signed && value >= 0 ? "+" : ""}${formatted}`;
}

export function AnimatedQuote({
  value,
  style,
  tone = "default",
  signed = false,
  fractionDigits = 2,
  suffix = "",
  durationMs = COUNT_UP_MS,
}: AnimatedQuoteProps) {
  const { tokens } = useTheme();
  const unavailable = value == null || !Number.isFinite(value);
  const target = unavailable ? 0 : (value as number);
  const [displayValue, setDisplayValue] = useState(target);
  const previousRef = useRef(target);
  const countRef = useRef(new Animated.Value(1)).current;
  const pulseRef = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (unavailable) {
      previousRef.current = 0;
      setDisplayValue(0);
      return;
    }
    const start = previousRef.current;
    if (start === target) {
      setDisplayValue(target);
      return;
    }
    previousRef.current = target;
    setDisplayValue(start);

    countRef.setValue(0);
    const listenerId = countRef.addListener(({ value: progress }) => {
      setDisplayValue(start + (target - start) * progress);
    });
    const countUp = Animated.timing(countRef, {
      toValue: 1,
      duration: durationMs,
      easing: Easing.out(Easing.cubic),
      useNativeDriver: false,
    });
    countUp.start(({ finished }) => {
      countRef.removeListener(listenerId);
      if (finished) setDisplayValue(target);
    });

    pulseRef.setValue(0);
    const pulse = Animated.sequence([
      Animated.timing(pulseRef, {
        toValue: 1,
        duration: PULSE_UP_MS,
        easing: Easing.out(Easing.quad),
        useNativeDriver: false,
      }),
      Animated.timing(pulseRef, {
        toValue: 0,
        duration: PULSE_DOWN_MS,
        easing: Easing.inOut(Easing.quad),
        useNativeDriver: false,
      }),
    ]);
    pulse.start();

    return () => {
      countUp.stop();
      countRef.removeListener(listenerId);
      pulse.stop();
    };
  }, [countRef, durationMs, pulseRef, target, unavailable]);

  if (unavailable) {
    return <Text style={style}>—</Text>;
  }

  const glowColor =
    tone === "positive" ? tokens.positive : tone === "negative" ? tokens.negative : tokens.primary;

  return (
    <Animated.View
      style={[
        styles.root,
        {
          shadowColor: glowColor,
          shadowOffset: { height: 0, width: 0 },
          shadowOpacity: pulseRef.interpolate({ inputRange: [0, 1], outputRange: [0, 0.65] }),
          shadowRadius: pulseRef.interpolate({ inputRange: [0, 1], outputRange: [0, 12] }),
          transform: [
            { scale: pulseRef.interpolate({ inputRange: [0, 1], outputRange: [1, 1.12] }) },
          ],
        },
      ]}
    >
      <Text style={style}>
        {`${formatNumber(displayValue, fractionDigits, signed)}${suffix}`}
      </Text>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  root: { alignSelf: "flex-start" },
});