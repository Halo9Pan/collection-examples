/*
 * Created on Sep 30, 2004
 */
package info.halo9pan.examples.prospring3.ch6;

import java.lang.reflect.Method;

import org.springframework.aop.ThrowsAdvice;
import org.springframework.aop.framework.ProxyFactory;

/**
 * @author clarence
 */
public class SimpleThrowsAdvice implements ThrowsAdvice {

	public static void main(String[] args) throws Exception {
		ErrorBean errorBean = new ErrorBean();

		ProxyFactory pf = new ProxyFactory();
		pf.setTarget(errorBean);
		pf.addAdvice(new SimpleThrowsAdvice());

		ErrorBean proxy = (ErrorBean) pf.getProxy();

		try {
			proxy.errorProneMethod();
		} catch (Exception ignored) {

		}

		try {
			proxy.otherErrorProneMethod();
		} catch (Exception ignored) {

		}

	}

	public void afterThrowing(Exception ex) throws Throwable {
		System.out.println("***");
		System.out.println("Generic Exception Capture");
		System.out.println("Caught: " + ex.getClass().getName());
		System.out.println("***\n");
	}

	public void afterThrowing(Method method, Object[] args, Object target, IllegalArgumentException ex)
			throws Throwable {
		System.out.println("***");
		System.out.println("IllegalArgumentException Capture");
		System.out.println("Caught: " + ex.getClass().getName());
		System.out.println("Method: " + method.getName());
		System.out.println("***\n");
	}
}